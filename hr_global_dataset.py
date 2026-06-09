#!/usr/bin/env python3
"""
hr_global_dataset.py

HDF5-backed dataset for training the global HR PARADIS model.

Reads pre-generated paired LR/HR data from swe_paired.h5 (produced by
generate_dataset.py) and returns (inp_fields, inp_winds, tar_fields, tar_winds)
tuples — identical format to PdeDatasetWithWinds — so SWELightningModule in
train.py works without modification.

HDF5 layout expected (written by generate_dataset.py):
    hr/t0  [num_ics, 3, hr_nlat, hr_nlon]  fields at t
    hr/t1  [num_ics, 3, hr_nlat, hr_nlon]  fields at t+dt
    hr/w0  [num_ics, 2, hr_nlat, hr_nlon]  winds  at t
    hr/w1  [num_ics, 2, hr_nlat, hr_nlon]  winds  at t+dt

Normalisation stats stored as HDF5 root attrs:
    hr_inp_mean  [3]   field channel means
    hr_inp_var   [3]   field channel variances
    hr_wind_mean [2]   wind  channel means
    hr_wind_var  [2]   wind  channel variances
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HRGlobalDataset(Dataset):
    """
    Global HR field dataset backed by swe_paired.h5.

    Parameters
    ----------
    h5_path : str
        Path to swe_paired.h5 produced by generate_dataset.py.
    split : str
        One of "train", "val", "test".
    num_train_ics : int
        Number of ICs reserved for training (= config["data"]["num_train_examples"]).
    num_val_ics : int
        Number of ICs reserved for validation (= config["data"]["num_val_examples"]).
    normalize : bool
        Apply per-channel normalisation using HDF5 attrs.
    preload : bool
        If True, load all IC arrays into RAM at construction time.
        Recommended for small datasets on Colab.
    """

    def __init__(
        self,
        h5_path: str,
        split: str,
        num_train_ics: int,
        num_val_ics: int,
        normalize: bool = True,
        preload: bool = False,
    ):
        assert split in ("train", "val", "test"), f"Unknown split '{split}'"
        self.h5_path = h5_path
        self.normalize = normalize
        self.preload = preload

        with h5py.File(h5_path, "r") as hf:
            total_ics = int(hf.attrs["num_ics"])

            # IC index ranges per split
            train_end = num_train_ics
            val_end   = num_train_ics + num_val_ics
            if split == "train":
                self._ic_slice = slice(0, train_end)
            elif split == "val":
                self._ic_slice = slice(train_end, val_end)
            else:  # test
                self._ic_slice = slice(val_end, total_ics)

            self.num_ics = len(range(*self._ic_slice.indices(total_ics)))

            # Normalisation stats
            def _t(k):
                return torch.tensor(np.array(hf.attrs[k]), dtype=torch.float32)

            self.f_mean = _t("hr_inp_mean").reshape(3, 1, 1)
            self.f_var  = _t("hr_inp_var" ).reshape(3, 1, 1)
            self.w_mean = _t("hr_wind_mean").reshape(2, 1, 1)
            self.w_var  = _t("hr_wind_var" ).reshape(2, 1, 1)

            if preload:
                self._cache = {
                    "hr_fields": torch.tensor(
                        np.array(hf["hr/fields"][self._ic_slice]), dtype=torch.float32
                    ),
                    "hr_winds": torch.tensor(
                        np.array(hf["hr/winds"][self._ic_slice]), dtype=torch.float32
                    ),
                }
            else:
                self._cache = None

    def __len__(self) -> int:
        return self.num_ics

    def __getitem__(self, idx: int):
        if self._cache is not None:
            hr_fields = self._cache["hr_fields"][idx]
            hr_winds = self._cache["hr_winds"][idx]
        else:
            start = self._ic_slice.start or 0
            global_idx = start + idx
            with h5py.File(self.h5_path, "r") as hf:
                hr_fields = torch.tensor(
                    np.array(hf["hr/fields"][global_idx]), dtype=torch.float32
                )
                hr_winds = torch.tensor(
                    np.array(hf["hr/winds"][global_idx]), dtype=torch.float32
                )

        inp_f = hr_fields[0]
        tar_f = hr_fields[1]
        inp_w = hr_winds[0]
        tar_w = hr_winds[1]

        if self.normalize:
            inp_f = (inp_f - self.f_mean) / self.f_var.sqrt()
            inp_w = (inp_w - self.w_mean) / self.w_var.sqrt()
            tar_f = (tar_f - self.f_mean) / self.f_var.sqrt()
            tar_w = (tar_w - self.w_mean) / self.w_var.sqrt()

        return inp_f, inp_w, tar_f, tar_w
