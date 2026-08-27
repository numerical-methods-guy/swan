"""
lam_patch_dataset.py

Patch dataset for the LAM workflow.

Reads from a pre-generated HDF5 file (made via generate_dataset.py) and yields
paired (hr_patch_t0, lr_halo_window_t0, hr_patch_target_t1) tuples.

Geometry (all sizes in *LR cells* unless noted)
------------------------------------------------
Given:
  - patch_nlat_lr, patch_nlon_lr   : interior patch size on the LR grid
  - halo_radius                    : uniform halo in LR cells on each side
  - s = upscale_factor_lat/lon     : HR = s × LR  (must be equal, i.e. isotropic)

  LR window = (patch_nlat_lr + 2*R) × (patch_nlon_lr + 2*R)   <- from LR global
  HR patch  = (patch_nlat_lr*s)    × (patch_nlon_lr*s)         <- from HR global

LR window includes LR footprint of the HR patch *plus* the R-cell halo
perimeter.  Every corner of HR patch therefore has at least 1 halo cell
diagonally outside it (as long as R >= 1).

Normalisation
-------------
All fields and winds normalised using the statistics stored in the HDF5
file (computed at generation time on 100 random ICs for each resolution).
Normalisation is:  x_norm = (x - mean) / sqrt(var)

Periodic boundary handling
---------------------------
Longitude wraps cyclically.  Latitude does NOT wrap (poles are excluded by
`exclude_pole_rows`).  If a window extends beyond ±90° in latitude the sample
is skipped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import ceil
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Patch indexing
# ---------------------------------------------------------------------------

@dataclass
class PatchEntry:
    ic_idx: int          # which IC (row in HDF5)
    lat0_lr: int         # top-left corner of the *interior* patch on LR grid
    lon0_lr: int         # (before halo is added)


def build_patch_manifest(
    num_ics: int,
    lr_nlat: int,
    lr_nlon: int,
    patch_nlat_lr: int,
    patch_nlon_lr: int,
    halo_radius: int,
    exclude_pole_rows: int,
) -> List[PatchEntry]:
    """Enumerate all valid patch centres for every IC.

    A patch is valid when the full LR window (patch + halo) fits within the
    non-polar latitude band.  Longitude wraps cyclically so lon is always valid.
    Patches are laid out on a non-overlapping grid; adjust strides below if you
    want overlapping patches for data augmentation.
    """
    entries: List[PatchEntry] = []

    # valid latitude range for the top-left corner of the interior patch
    lat_min = exclude_pole_rows + halo_radius
    lat_max = lr_nlat - exclude_pole_rows - halo_radius - patch_nlat_lr

    # stride = patch size (non-overlapping tiling)
    lat_stride = patch_nlat_lr
    lon_stride = patch_nlon_lr

    for ic in range(num_ics):
        lat0 = lat_min
        while lat0 + patch_nlat_lr <= lat_max + patch_nlat_lr:  # inclusive upper
            if lat0 > lat_max:
                break
            for lon0 in range(0, lr_nlon, lon_stride):
                entries.append(PatchEntry(ic_idx=ic, lat0_lr=lat0, lon0_lr=lon0))
            lat0 += lat_stride

    return entries

def build_patch_plan_tensors(
    lr_nlat: int,
    lr_nlon: int,
    patch_nlat_lr: int,
    patch_nlon_lr: int,
    halo_radius: int,
    upscale_factor: int,
    exclude_pole_rows: int = 4,
):
    R = halo_radius
    s = upscale_factor

    win_nlat = patch_nlat_lr + 2 * R
    win_nlon = patch_nlon_lr + 2 * R
    patch_nlat_hr = patch_nlat_lr * s
    patch_nlon_hr = patch_nlon_lr * s

    lat_min = exclude_pole_rows + R
    lat_max = lr_nlat - exclude_pole_rows - R - patch_nlat_lr

    rows = []
    lat0 = lat_min
    while lat0 + patch_nlat_lr <= lat_max + patch_nlat_lr:
        if lat0 > lat_max:
            break
        for lon0 in range(0, lr_nlon, patch_nlon_lr):
            rows.append((
                lat0,
                lon0,
                lat0 - R,
                lon0 - R,
                lat0 * s,
                lon0 * s,
            ))
        lat0 += patch_nlat_lr

    if not rows:
        raise ValueError(
            "No valid inference patches were generated. "
            "Check patch size / halo / exclude_pole_rows / grid dimensions."
        )

    arr = torch.tensor(rows, dtype=torch.long)
    return {
        "lat0_lr": arr[:, 0],
        "lon0_lr": arr[:, 1],
        "win_lat0": arr[:, 2],
        "win_lon0": arr[:, 3],
        "hr_lat0": arr[:, 4],
        "hr_lon0": arr[:, 5],
        "num_patches": int(arr.shape[0]),
        "win_nlat": int(win_nlat),
        "win_nlon": int(win_nlon),
        "patch_nlat_hr": int(patch_nlat_hr),
        "patch_nlon_hr": int(patch_nlon_hr),
    }

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LAMPatchDataset(Dataset):
    """
    Optimized patch dataset for the LAM workflow.

    Main changes:
    - caches normalization std tensors once in __init__
    - removes duplicated split logic
    - uses a worker-local lazy HDF5 handle when preload=False
    - keeps optional split-local RAM preload
    - precomputes patch coordinate arrays used in __getitem__
    """

    def __init__(
        self,
        h5_path: str,
        patch_nlat_lr: int,
        patch_nlon_lr: int,
        halo_radius: int,
        exclude_pole_rows: int = 4,
        split: str = "train",
        num_train_ics: int | None = None,
        normalize: bool = True,
        preload: bool = False,
        lr_to_hr_interp: str = "bilinear",
        max_rollout_steps: int = 1,
        finetuning_enabled: bool = False,
    ):
        super().__init__()
        assert split in ("train", "val", "test")
        assert halo_radius >= 1, "halo_radius must be >= 1"

        self.h5_path = h5_path
        self.patch_nlat_lr = int(patch_nlat_lr)
        self.patch_nlon_lr = int(patch_nlon_lr)
        self.halo_radius = int(halo_radius)
        self.exclude_pole_rows = int(exclude_pole_rows)
        self.normalize = bool(normalize)
        self.preload = bool(preload)
        self.lr_to_hr_interp = str(lr_to_hr_interp)
        self.max_rollout_steps = int(max_rollout_steps)
        self.finetuning_enabled = bool(finetuning_enabled)
        if self.max_rollout_steps < 1:
            raise ValueError(
                f"max_rollout_steps must be >= 1, got {self.max_rollout_steps}"
            )

        with h5py.File(h5_path, "r") as hf:
            self.lr_nlat = int(hf.attrs["lr_nlat"])
            self.lr_nlon = int(hf.attrs["lr_nlon"])
            self.hr_nlat = int(hf.attrs["hr_nlat"])
            self.hr_nlon = int(hf.attrs["hr_nlon"])
            self.s_lat = int(hf.attrs["upscale_factor_lat"])
            self.s_lon = int(hf.attrs["upscale_factor_lon"])
            total_ics = int(hf.attrs["num_ics"])

            self.rollout_steps_available = int(hf.attrs["rollout_steps"])
            if self.max_rollout_steps > self.rollout_steps_available:
                raise ValueError(
                    "Requested max_rollout_steps="
                    f"{self.max_rollout_steps}, but HDF5 stores only "
                    f"{self.rollout_steps_available} forecast steps."
                )

            def _t(key):
                return torch.tensor(np.array(hf.attrs[key]), dtype=torch.float32)

            self.lr_inp_mean = _t("lr_inp_mean").reshape(3, 1, 1)
            self.lr_inp_var = _t("lr_inp_var").reshape(3, 1, 1)
            self.lr_wind_mean = _t("lr_wind_mean").reshape(2, 1, 1)
            self.lr_wind_var = _t("lr_wind_var").reshape(2, 1, 1)
            self.hr_inp_mean = _t("hr_inp_mean").reshape(3, 1, 1)
            self.hr_inp_var = _t("hr_inp_var").reshape(3, 1, 1)
            self.hr_wind_mean = _t("hr_wind_mean").reshape(2, 1, 1)
            self.hr_wind_var = _t("hr_wind_var").reshape(2, 1, 1)

        assert self.s_lat == self.s_lon, (
            f"Non-isotropic upscale factors ({self.s_lat} lat, {self.s_lon} lon) "
            "are not currently supported. Set refinement_factor_lat == refinement_factor_lon."
        )

        self.s = self.s_lat
        self.patch_nlat_hr = self.patch_nlat_lr * self.s
        self.patch_nlon_hr = self.patch_nlon_lr * self.s
        self.win_nlat = self.patch_nlat_lr + 2 * self.halo_radius
        self.win_nlon = self.patch_nlon_lr + 2 * self.halo_radius
        # True / 1: valid exterior LR halo data.
        # False / 0: central LR footprint aligned with the HR patch.
        lr_halo_valid_mask = torch.ones(
            1,
            self.win_nlat,
            self.win_nlon,
            dtype=torch.float32,
        )

        row_start = self.halo_radius
        row_end = row_start + self.patch_nlat_lr
        col_start = self.halo_radius
        col_end = col_start + self.patch_nlon_lr

        lr_halo_valid_mask[:, row_start:row_end, col_start:col_end] = 0.0

        self.lr_halo_valid_mask = lr_halo_valid_mask

        self.lr_inp_std = self.lr_inp_var.sqrt()
        self.lr_wind_std = self.lr_wind_var.sqrt()
        self.hr_inp_std = self.hr_inp_var.sqrt()
        self.hr_wind_std = self.hr_wind_var.sqrt()

        if num_train_ics is None:
            num_train_ics = int(round(0.8 * total_ics))
            num_train_ics = min(num_train_ics, total_ics)

        num_val_ics = total_ics - num_train_ics

        if split == "train":
            self._ic_offset = 0
            self._num_ics = num_train_ics
        elif split == "val":
            self._ic_offset = num_train_ics
            self._num_ics = num_val_ics
        else:
            self._ic_offset = num_train_ics
            self._num_ics = num_val_ics

        assert self._num_ics > 0, (
            f"No ICs for split='{split}' (total={total_ics}, num_train={num_train_ics})"
        )

        self._manifest = build_patch_manifest(
            num_ics=self._num_ics,
            lr_nlat=self.lr_nlat,
            lr_nlon=self.lr_nlon,
            patch_nlat_lr=self.patch_nlat_lr,
            patch_nlon_lr=self.patch_nlon_lr,
            halo_radius=self.halo_radius,
            exclude_pole_rows=self.exclude_pole_rows,
        )

        for e in self._manifest:
            e.ic_idx += self._ic_offset

        self._num_samples = len(self._manifest)

        self._ic_indices = np.asarray([e.ic_idx for e in self._manifest], dtype=np.int64)
        self._lat0_lr = np.asarray([e.lat0_lr for e in self._manifest], dtype=np.int64)
        self._lon0_lr = np.asarray([e.lon0_lr for e in self._manifest], dtype=np.int64)
        self._win_lat0 = self._lat0_lr - self.halo_radius
        self._win_lon0 = self._lon0_lr - self.halo_radius
        self._hr_lat0 = self._lat0_lr * self.s
        self._hr_lon0 = self._lon0_lr * self.s

        self._cache: dict | None = None
        self._hf = None
        self._last_ic_idx = None
        self._last_ic_data = None

        if self.preload:
            self._preload_data()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_hf"] = None
        state["_last_ic_idx"] = None
        state["_last_ic_data"] = None
        return state

    def __del__(self):
        hf = getattr(self, "_hf", None)
        if hf is not None:
            try:
                hf.close()
            except Exception:
                pass

    def _ensure_h5_open(self):
        if self._hf is None:
            self._hf = h5py.File(self.h5_path, "r")

    def _preload_data(self):
        """Load only the ICs for this split into CPU RAM."""
        ic_slice = slice(self._ic_offset, self._ic_offset + self._num_ics)
        with h5py.File(self.h5_path, "r") as hf:
            self._cache = {
                "lr_fields": torch.tensor(np.array(hf["lr/fields"][ic_slice]), dtype=torch.float32),
                "lr_winds": torch.tensor(np.array(hf["lr/winds"][ic_slice]), dtype=torch.float32),
                "hr_fields": torch.tensor(np.array(hf["hr/fields"][ic_slice]), dtype=torch.float32),
                "hr_winds": torch.tensor(np.array(hf["hr/winds"][ic_slice]), dtype=torch.float32),
            }

    def _load_ic(self, ic_idx: int):
        """Return (lr_fields, lr_winds, hr_fields, hr_winds) for one absolute IC index."""
        if self._cache is not None:
            local = ic_idx - self._ic_offset
            return (
                self._cache["lr_fields"][local],
                self._cache["lr_winds"][local],
                self._cache["hr_fields"][local],
                self._cache["hr_winds"][local],
            )

        if self._last_ic_idx == ic_idx and self._last_ic_data is not None:
            return self._last_ic_data

        self._ensure_h5_open()
        hf = self._hf

        data = (
            torch.from_numpy(np.asarray(hf["lr/fields"][ic_idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(hf["lr/winds"][ic_idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(hf["hr/fields"][ic_idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(hf["hr/winds"][ic_idx], dtype=np.float32)),
        )

        self._last_ic_idx = ic_idx
        self._last_ic_data = data
        return data

    @staticmethod
    def _crop_lon_cyclic(tensor: torch.Tensor, lon0: int, width: int) -> torch.Tensor:
        """Crop `width` columns starting at `lon0` with cyclic longitude."""
        nlon = tensor.shape[-1]
        lon0 = lon0 % nlon
        if lon0 + width <= nlon:
            return tensor[..., lon0: lon0 + width]
        part1 = tensor[..., lon0:]
        part2 = tensor[..., : (lon0 + width) % nlon]
        return torch.cat([part1, part2], dim=-1)

    def __len__(self) -> int:
        return self._num_samples

    def _upsample_lr_patch_to_hr(
        self,
        lr_patch: torch.Tensor,
    ) -> torch.Tensor:
        if self.lr_to_hr_interp == "nearest":
            return F.interpolate(
                lr_patch.unsqueeze(0),
                size=(self.patch_nlat_hr, self.patch_nlon_hr),
                mode="nearest",
            ).squeeze(0)

        return F.interpolate(
            lr_patch.unsqueeze(0),
            size=(self.patch_nlat_hr, self.patch_nlon_hr),
            mode=self.lr_to_hr_interp,
            align_corners=False,
        ).squeeze(0)


    def _normalize_lr_state(self, fields, winds):
        return torch.cat(
            [
                (fields - self.lr_inp_mean) / self.lr_inp_std,
                (winds - self.lr_wind_mean) / self.lr_wind_std,
            ],
            dim=0,
        )


    def _normalize_hr_state(self, fields, winds):
        return torch.cat(
            [
                (fields - self.hr_inp_mean) / self.hr_inp_std,
                (winds - self.hr_wind_mean) / self.hr_wind_std,
            ],
            dim=0,
        )

    def __getitem__(self, idx: int) -> dict:
        ic_idx = int(self._ic_indices[idx])
        lat0_lr = int(self._lat0_lr[idx])
        lon0_lr = int(self._lon0_lr[idx])
        win_lat0 = int(self._win_lat0[idx])
        win_lon0 = int(self._win_lon0[idx])
        hr_lat0 = int(self._hr_lat0[idx])
        hr_lon0 = int(self._hr_lon0[idx])

        lr_fields, lr_winds, hr_fields, hr_winds = self._load_ic(ic_idx)

        lr_halos = []
        lr_inputs_hr = []
        hr_targets = []
        lr_refs_hr = []

        for lead_idx in range(self.max_rollout_steps):
            curr_t = lead_idx
            next_t = lead_idx + 1

            # Five-channel LR halo input at current time t.
            lr_halo_f = lr_fields[
                curr_t,
                :,
                win_lat0: win_lat0 + self.win_nlat,
                :,
            ]
            lr_halo_w = lr_winds[
                curr_t,
                :,
                win_lat0: win_lat0 + self.win_nlat,
                :,
            ]

            lr_halo_f = self._crop_lon_cyclic(
                lr_halo_f,
                win_lon0,
                self.win_nlon,
            )
            lr_halo_w = self._crop_lon_cyclic(
                lr_halo_w,
                win_lon0,
                self.win_nlon,
            )

            lr_halo = (
                self._normalize_lr_state(lr_halo_f, lr_halo_w)
                if self.normalize
                else torch.cat([lr_halo_f, lr_halo_w], dim=0)
            )

            # Exclude LR values geographically aligned with the HR center.
            # The exterior LR ring remains available as boundary forcing.
            lr_halo = lr_halo * self.lr_halo_valid_mask

            lr_halos.append(lr_halo)

            # LR state at current time t, cropped to the interior patch and
            # interpolated to HR resolution. This is used to blend the HR input
            # before it enters the HR encoder.
            lr_input_f = lr_fields[
                curr_t,
                :,
                lat0_lr: lat0_lr + self.patch_nlat_lr,
                :,
            ]
            lr_input_w = lr_winds[
                curr_t,
                :,
                lat0_lr: lat0_lr + self.patch_nlat_lr,
                :,
            ]

            lr_input_f = self._crop_lon_cyclic(
                lr_input_f,
                lon0_lr,
                self.patch_nlon_lr,
            )
            lr_input_w = self._crop_lon_cyclic(
                lr_input_w,
                lon0_lr,
                self.patch_nlon_lr,
            )

            lr_input_f_hr = self._upsample_lr_patch_to_hr(lr_input_f)
            lr_input_w_hr = self._upsample_lr_patch_to_hr(lr_input_w)

            # Normalize with HR statistics, because this tensor will be blended
            # directly with the normalized HR patch passed into the HR encoder.
            lr_inputs_hr.append(
                self._normalize_hr_state(lr_input_f_hr, lr_input_w_hr)
                if self.normalize
                else torch.cat([lr_input_f_hr, lr_input_w_hr], dim=0)
            )

            # Five-channel HR truth target at t+1.
            hr_target_f = hr_fields[
                next_t,
                :,
                hr_lat0: hr_lat0 + self.patch_nlat_hr,
                :,
            ]
            hr_target_w = hr_winds[
                next_t,
                :,
                hr_lat0: hr_lat0 + self.patch_nlat_hr,
                :,
            ]

            hr_target_f = self._crop_lon_cyclic(
                hr_target_f,
                hr_lon0,
                self.patch_nlon_hr,
            )
            hr_target_w = self._crop_lon_cyclic(
                hr_target_w,
                hr_lon0,
                self.patch_nlon_hr,
            )

            hr_targets.append(
                self._normalize_hr_state(hr_target_f, hr_target_w)
                if self.normalize
                else torch.cat([hr_target_f, hr_target_w], dim=0)
            )

            # LR state at t+1, cropped to the interior and interpolated to HR.
            lr_ref_f = lr_fields[
                next_t,
                :,
                lat0_lr: lat0_lr + self.patch_nlat_lr,
                :,
            ]
            lr_ref_w = lr_winds[
                next_t,
                :,
                lat0_lr: lat0_lr + self.patch_nlat_lr,
                :,
            ]

            lr_ref_f = self._crop_lon_cyclic(
                lr_ref_f,
                lon0_lr,
                self.patch_nlon_lr,
            )
            lr_ref_w = self._crop_lon_cyclic(
                lr_ref_w,
                lon0_lr,
                self.patch_nlon_lr,
            )

            lr_ref_f_hr = self._upsample_lr_patch_to_hr(lr_ref_f)
            lr_ref_w_hr = self._upsample_lr_patch_to_hr(lr_ref_w)

            lr_refs_hr.append(
                self._normalize_hr_state(lr_ref_f_hr, lr_ref_w_hr)
                if self.normalize
                else torch.cat([lr_ref_f_hr, lr_ref_w_hr], dim=0)
            )

        # Five-channel initial HR state at t=0.
        hr_t0_f = hr_fields[
            0,
            :,
            hr_lat0: hr_lat0 + self.patch_nlat_hr,
            :,
        ]
        hr_t0_w = hr_winds[
            0,
            :,
            hr_lat0: hr_lat0 + self.patch_nlat_hr,
            :,
        ]

        hr_t0_f = self._crop_lon_cyclic(
            hr_t0_f,
            hr_lon0,
            self.patch_nlon_hr,
        )
        hr_t0_w = self._crop_lon_cyclic(
            hr_t0_w,
            hr_lon0,
            self.patch_nlon_hr,
        )

        hr_patch_t0 = (
            self._normalize_hr_state(hr_t0_f, hr_t0_w)
            if self.normalize
            else torch.cat([hr_t0_f, hr_t0_w], dim=0)
        )

        if not self.finetuning_enabled:
            return {
                "lr_halo": lr_halos[0],
                "hr_patch_t0": hr_patch_t0,
                "lr_patch_t0_hrref": lr_inputs_hr[0],
                "hr_patch_t1": hr_targets[0],
                "lr_patch_t1_hrref": lr_refs_hr[0],
                "meta": {
                    "ic_idx": ic_idx,
                    "lat0_lr": lat0_lr,
                    "lon0_lr": lon0_lr,
                },
            }

        return {
            "lr_halo_seq": torch.stack(lr_halos, dim=0),
            "hr_patch_t0": hr_patch_t0,
            "lr_input_seq": torch.stack(lr_inputs_hr, dim=0),
            "hr_target_seq": torch.stack(hr_targets, dim=0),
            "lr_ref_seq": torch.stack(lr_refs_hr, dim=0),
            "meta": {
                "ic_idx": ic_idx,
                "lat0_lr": lat0_lr,
                "lon0_lr": lon0_lr,
            },
        }

    def geometry_summary(self) -> str:
        R = self.halo_radius
        pL = self.patch_nlat_lr
        pN = self.patch_nlon_lr
        s = self.s
        lines = [
            "LAMPatchDataset geometry",
            f" LR grid : {self.lr_nlat} x {self.lr_nlon}",
            f" HR grid : {self.hr_nlat} x {self.hr_nlon}",
            f" Upscale factor : {s}x",
            f" Patch (LR cells) : {pL} lat x {pN} lon",
            f" Patch (HR cells) : {pL*s} lat x {pN*s} lon",
            f" Halo radius (LR) : {R} cells",
            f" LR window total : {pL+2*R} lat x {pN+2*R} lon",
            f" lr_halo tensor shape : [5, {pL+2*R}, {pN+2*R}]",
            f" hr_patch shape : [5, {pL*s}, {pN*s}]",
            f" Total patches : {len(self)}",
        ]
        return "\n".join(lines)