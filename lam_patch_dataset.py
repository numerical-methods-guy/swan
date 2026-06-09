"""
lam_patch_dataset.py

Patch dataset for the LAM workflow.

Reads from a pre-generated HDF5 file (see generate_dataset.py) and yields
paired (hr_patch_t0, lr_halo_window_t0, hr_patch_target_t1) tuples.

Geometry (all sizes in *LR cells* unless noted)
------------------------------------------------
Given:
  - patch_nlat_lr, patch_nlon_lr   : interior patch size on the LR grid
  - halo_radius                    : uniform halo in LR cells on each side
  - s = upscale_factor_lat/lon     : HR = s × LR  (must be equal, i.e. isotropic)

  LR window = (patch_nlat_lr + 2*R) × (patch_nlon_lr + 2*R)   <- from LR global
  HR patch  = (patch_nlat_lr*s)    × (patch_nlon_lr*s)         <- from HR global

The LR window includes the LR footprint of the HR patch *plus* the R-cell halo
perimeter.  Every corner of the HR patch therefore has at least 1 halo cell
diagonally outside it (as long as R >= 1).

Normalisation
-------------
All fields and winds are normalised using the statistics stored in the HDF5
file (computed at generation time on 200 random ICs for each resolution).
Normalisation is:  x_norm = (x - mean) / sqrt(var)

Periodic boundary handling
---------------------------
Longitude wraps cyclically.  Latitude does NOT wrap (poles are excluded by
`exclude_pole_rows`).  If a window extends beyond ±90° in latitude the sample
is skipped in the manifest.
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
# Patch manifest
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LAMPatchDataset(Dataset):
    """
    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file produced by generate_dataset.py.
    patch_nlat_lr, patch_nlon_lr : int
        Interior patch size in LR cells.
    halo_radius : int
        Uniform halo thickness in LR cells on each side of the patch.
    exclude_pole_rows : int
        Do not place patch centres in the top/bottom N LR rows.
    split : "train" | "val"
        Train uses ICs [0, num_train), val uses [num_train, num_ics).
    num_train_ics : int
        Number of ICs allocated to training (remainder go to val).
    normalize : bool
        Apply per-channel normalisation using stats from the HDF5 file.
    preload : bool
        If True, load the entire HDF5 file into RAM at init time (fast but
        memory-hungry).  If False, read slices on the fly (slower, lower RAM).
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
    ):
        super().__init__()
        assert split in ("train", "val", "test")

        assert halo_radius >= 1, "halo_radius must be >= 1 to satisfy diagonal-corner requirement"

        self.h5_path       = h5_path
        self.patch_nlat_lr = patch_nlat_lr
        self.patch_nlon_lr = patch_nlon_lr
        self.halo_radius   = halo_radius
        self.normalize     = normalize
        self.preload       = preload

        # --- read metadata from HDF5 -------------------------------------------
        with h5py.File(h5_path, "r") as hf:
            self.lr_nlat   = int(hf.attrs["lr_nlat"])
            self.lr_nlon   = int(hf.attrs["lr_nlon"])
            self.hr_nlat   = int(hf.attrs["hr_nlat"])
            self.hr_nlon   = int(hf.attrs["hr_nlon"])
            self.s_lat     = int(hf.attrs["upscale_factor_lat"])
            self.s_lon     = int(hf.attrs["upscale_factor_lon"])
            total_ics      = int(hf.attrs["num_ics"])

            # normalisation stats — shape [C] stored as 1-D arrays
            def _t(key): return torch.tensor(np.array(hf.attrs[key]), dtype=torch.float32)
            self.lr_inp_mean  = _t("lr_inp_mean").reshape(3, 1, 1)
            self.lr_inp_var   = _t("lr_inp_var" ).reshape(3, 1, 1)
            self.lr_wind_mean = _t("lr_wind_mean").reshape(2, 1, 1)
            self.lr_wind_var  = _t("lr_wind_var" ).reshape(2, 1, 1)
            self.hr_inp_mean  = _t("hr_inp_mean" ).reshape(3, 1, 1)
            self.hr_inp_var   = _t("hr_inp_var"  ).reshape(3, 1, 1)
            self.hr_wind_mean = _t("hr_wind_mean").reshape(2, 1, 1)
            self.hr_wind_var  = _t("hr_wind_var" ).reshape(2, 1, 1)

        assert self.s_lat == self.s_lon, (
            f"Non-isotropic upscale factors ({self.s_lat} lat, {self.s_lon} lon) are not "
            "currently supported.  Set refinement_factor_lat == refinement_factor_lon."
        )
        self.s = self.s_lat

        # derived HR patch size
        self.patch_nlat_hr = patch_nlat_lr * self.s
        self.patch_nlon_hr = patch_nlon_lr * self.s

        # --- IC split -----------------------------------------------------------
        if num_train_ics is None:
            num_train_ics = int(round(0.8 * total_ics))
        num_train_ics = min(num_train_ics, total_ics)

        num_val_ics = total_ics - num_train_ics   # all remaining ICs go to val/test

        if split == "train":
            self._ic_offset = 0
            self._num_ics   = num_train_ics
        elif split == "val":
            self._ic_offset = num_train_ics
            self._num_ics   = num_val_ics
        else:  # "test"
            self._ic_offset = num_train_ics
            self._num_ics   = num_val_ics

        # --- IC split (needs total_ics from HDF5 read above) -------------------
        if num_train_ics is None:
            num_train_ics = int(round(0.8 * total_ics))
        num_train_ics = min(num_train_ics, total_ics)
        num_val_ics   = total_ics - num_train_ics

        if split == "train":
            self._ic_offset = 0
            self._num_ics   = num_train_ics
        elif split == "val":
            self._ic_offset = num_train_ics
            self._num_ics   = num_val_ics
        else:  # test
            self._ic_offset = num_train_ics
            self._num_ics   = num_val_ics

        assert self._num_ics > 0, (
            f"No ICs for split='{split}' (total={total_ics}, num_train={num_train_ics})"
        )

        # --- build patch manifest -----------------------------------------------
        self._manifest = build_patch_manifest(
            num_ics           = self._num_ics,
            lr_nlat           = self.lr_nlat,
            lr_nlon           = self.lr_nlon,
            patch_nlat_lr     = patch_nlat_lr,
            patch_nlon_lr     = patch_nlon_lr,
            halo_radius       = halo_radius,
            exclude_pole_rows = exclude_pole_rows,
        )
        # remap ic_idx to absolute HDF5 row
        for e in self._manifest:
            e.ic_idx += self._ic_offset

        # --- optional preload ---------------------------------------------------
        self._cache: dict | None = None
        if preload:
            self._preload_data()

    # ------------------------------------------------------------------
    # Preload
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_ic(self, ic_idx: int) -> Tuple[torch.Tensor, ...]:
        """Return (lr_fields, lr_winds, hr_fields, hr_winds) for one absolute IC index."""
        if self._cache is not None:
            local = ic_idx - self._ic_offset
            return (
                self._cache["lr_fields"][local],
                self._cache["lr_winds"][local],
                self._cache["hr_fields"][local],
                self._cache["hr_winds"][local],
            )

        with h5py.File(self.h5_path, "r") as hf:
            lr_fields = torch.tensor(np.array(hf["lr/fields"][ic_idx]), dtype=torch.float32)
            lr_winds = torch.tensor(np.array(hf["lr/winds"][ic_idx]), dtype=torch.float32)
            hr_fields = torch.tensor(np.array(hf["hr/fields"][ic_idx]), dtype=torch.float32)
            hr_winds = torch.tensor(np.array(hf["hr/winds"][ic_idx]), dtype=torch.float32)

        return lr_fields, lr_winds, hr_fields, hr_winds
    
    @staticmethod
    def _crop_lon_cyclic(tensor: torch.Tensor, lon0: int, width: int) -> torch.Tensor:
        """Crop `width` columns starting at `lon0` with cyclic (periodic) longitude.

        tensor : [..., nlat, nlon]
        Returns : [..., nlat, width]
        """
        nlon = tensor.shape[-1]
        lon0 = lon0 % nlon
        if lon0 + width <= nlon:
            return tensor[..., lon0 : lon0 + width]
        # wrap-around: stitch two slices
        part1 = tensor[..., lon0:]
        part2 = tensor[..., : (lon0 + width) % nlon]
        return torch.cat([part1, part2], dim=-1)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._manifest)

    def __getitem__(self, idx: int) -> dict:
        entry = self._manifest[idx]
        lr_fields, lr_winds, hr_fields, hr_winds = self._load_ic(entry.ic_idx)

        lr_t0 = lr_fields[0]
        lr_w0 = lr_winds[0]
        hr_t0 = hr_fields[0]
        hr_t1 = hr_fields[1]
        hr_w0 = hr_winds[0]

        R  = self.halo_radius
        pL = self.patch_nlat_lr
        pN = self.patch_nlon_lr
        s  = self.s

        # --- LR halo window (fields + winds) ------------------------------------
        # Top-left of the window (including halo) on the LR grid
        win_lat0 = entry.lat0_lr - R
        win_lon0 = entry.lon0_lr - R   # lon wraps; lat is guaranteed valid by manifest
        win_nlat = pL + 2 * R
        win_nlon = pN + 2 * R

        # latitude slice (always in-bounds by manifest construction)
        lr_halo_f = lr_t0[:, win_lat0 : win_lat0 + win_nlat, :]   # [..., win_nlat, lr_nlon]
        lr_halo_w = lr_w0[:, win_lat0 : win_lat0 + win_nlat, :]
        # cyclic longitude crop
        lr_halo_f = self._crop_lon_cyclic(lr_halo_f, win_lon0, win_nlon)  # [3, win_nlat, win_nlon]
        lr_halo_w = self._crop_lon_cyclic(lr_halo_w, win_lon0, win_nlon)  # [2, win_nlat, win_nlon]

        # --- HR patch interior at t (input) and t+dt (target) -------------------
        # The HR interior patch corresponds to LR cells [lat0_lr, lat0_lr+pL) x [lon0_lr, lon0_lr+pN)
        hr_lat0 = entry.lat0_lr * s
        hr_lon0 = entry.lon0_lr * s

        hr_patch_w0 = hr_w0[:, hr_lat0 : hr_lat0 + self.patch_nlat_hr, :]
        hr_patch_w0 = self._crop_lon_cyclic(hr_patch_w0, hr_lon0, self.patch_nlon_hr)

        hr_patch_t0 = hr_t0[:, hr_lat0 : hr_lat0 + self.patch_nlat_hr, :]
        hr_patch_t0 = self._crop_lon_cyclic(hr_patch_t0, hr_lon0, self.patch_nlon_hr)  # [3, Hhr, Whr]

        hr_patch_t1 = hr_t1[:, hr_lat0 : hr_lat0 + self.patch_nlat_hr, :]
        hr_patch_t1 = self._crop_lon_cyclic(hr_patch_t1, hr_lon0, self.patch_nlon_hr)  # [3, Hhr, Whr]

        # --- Normalisation ------------------------------------------------------
        if self.normalize:
            lr_halo_f   = (lr_halo_f   - self.lr_inp_mean)  / self.lr_inp_var.sqrt()
            lr_halo_w   = (lr_halo_w   - self.lr_wind_mean) / self.lr_wind_var.sqrt()
            hr_patch_t0 = (hr_patch_t0 - self.hr_inp_mean)  / self.hr_inp_var.sqrt()
            hr_patch_t1 = (hr_patch_t1 - self.hr_inp_mean)  / self.hr_inp_var.sqrt()
            hr_patch_w0 = (hr_patch_w0 - self.hr_wind_mean) / self.hr_wind_var.sqrt()

        return {
            # LR halo window at t  — shape [5, win_nlat, win_nlon]  (fields + winds)
            "lr_halo":      torch.cat([lr_halo_f, lr_halo_w], dim=0),
            # HR patch interior at t — shape [5, patch_nlat_hr, patch_nlon_hr] (fields + winds)
            "hr_patch_t0": torch.cat([hr_patch_t0, hr_patch_w0], dim=0),
            # HR patch interior at t+dt — shape [3, patch_nlat_hr, patch_nlon_hr]
            "hr_patch_t1":  hr_patch_t1,
            # convenience: patch position info (for diagnostics / visualisation)
            "meta": {
                "ic_idx":    entry.ic_idx,
                "lat0_lr":   entry.lat0_lr,
                "lon0_lr":   entry.lon0_lr,
            },
        }

    # ------------------------------------------------------------------
    # Geometry summary (useful for sanity-checking)
    # ------------------------------------------------------------------

    def geometry_summary(self) -> str:
        R  = self.halo_radius
        pL = self.patch_nlat_lr
        pN = self.patch_nlon_lr
        s  = self.s
        lines = [
            "LAMPatchDataset geometry",
            f"  LR grid              : {self.lr_nlat} x {self.lr_nlon}",
            f"  HR grid              : {self.hr_nlat} x {self.hr_nlon}",
            f"  Upscale factor       : {s}x",
            f"  Patch (LR cells)     : {pL} lat x {pN} lon",
            f"  Patch (HR cells)     : {pL*s} lat x {pN*s} lon",
            f"  Halo radius (LR)     : {R} cells",
            f"  LR window total      : {pL+2*R} lat x {pN+2*R} lon",
            f"  lr_halo tensor shape : [5, {pL+2*R}, {pN+2*R}]",
            f"  hr_patch shape       : [5, {pL*s}, {pN*s}]",
            f"  Total patches        : {len(self)}",
        ]
        return "\n".join(lines)
