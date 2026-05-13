# ignore_header_test
# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import os
import torch

from math import ceil

from dataset.shallow_water_solver import ShallowWaterSolver


class ShallowWaterPDEDataset(torch.utils.data.Dataset):
    """Custom Dataset class generating trainig data corresponding to
    the underlying PDEs of the Shallow Water Equations"""

    def __init__(
        self,
        dt,
        nsteps,
        dims=(384, 768),
        initial_condition="random",
        num_examples=32,
        device=torch.device("cpu"),
        normalize=True,
        rank=0,
        stream=None,
        dtype=torch.float32,
        precomputed_folder=None,
    ):
        self.dtype = dtype

        self.num_examples = num_examples
        self.device = device
        self.stream = stream
        self.rank = rank
        self.precomputed_folder = precomputed_folder

        self.nlat = dims[0]
        self.nlon = dims[1]

        # number of solver steps used to compute the target
        self.nsteps = nsteps
        self.normalize = normalize

        lmax = ceil(self.nlat / 3)
        mmax = lmax
        dt_solver = dt / float(self.nsteps)
        self.solver = (
            ShallowWaterSolver(
                self.nlat,
                self.nlon,
                dt_solver,
                lmax=lmax,
                mmax=mmax,
                grid="equiangular",
            )
            .to(self.device)
            .float()
        )

        self.set_initial_condition(ictype=initial_condition)

        inp0, tar0 = self._get_sample()
        self.inp_shape = inp0.shape
        self.tar_shape = tar0.shape

        if self.normalize:
            self.inp_mean = torch.mean(inp0, dim=(-1, -2)).reshape(-1, 1, 1)
            self.inp_var = torch.var(inp0, dim=(-1, -2)).reshape(-1, 1, 1)

    def __len__(self):
        length = self.num_examples if self.ictype in ("random", "precomputed") else 1
        return length

    def set_initial_condition(self, ictype="random", precomputed_folder=None):
        self.ictype = ictype
        if ictype == "precomputed":
            if precomputed_folder is None and self.precomputed_folder is None:
                raise ValueError("precomputed_folder must be provided when ictype='precomputed'")
            if precomputed_folder is not None:
                self.precomputed_folder = precomputed_folder

    def set_num_examples(self, num_examples=32):
        self.num_examples = num_examples

    def _get_sample(self, index=None):
        if self.ictype == "random":
            inp_spec = self.solver.random_initial_condition(mach=0.2)
            tar_spec = self.solver.timestep(inp_spec, self.nsteps)
        elif self.ictype == "galewsky":
            inp_spec = self.solver.galewsky_initial_condition()
            tar_spec = self.solver.timestep(inp_spec, self.nsteps)
        elif self.ictype == "precomputed":
            if index is None:
                raise ValueError("index must be provided when ictype='precomputed'")
            inp_spec = self.solver.precomputed_initial_condition(self.precomputed_folder, index, step=0)
            target_path = os.path.join(self.precomputed_folder, f"{index}_1.pt")
            if os.path.exists(target_path):
                tar_spec = self.solver.precomputed_initial_condition(self.precomputed_folder, index, step=1)
            else:
                tar_spec = self.solver.timestep(inp_spec, self.nsteps)
        else:
            raise NotImplementedError(
                f"Initial Condition {self.ictype} not implemented."
            )

        inp = self.solver.spec2grid(inp_spec)
        tar = self.solver.spec2grid(tar_spec)

        return inp, tar

    def __getitem__(self, index):
        if self.rank == 0:
            with torch.inference_mode():
                with torch.no_grad():
                    inp, tar = self._get_sample(index=index)

                    if self.normalize:
                        inp = (inp - self.inp_mean) / torch.sqrt(self.inp_var)
                        tar = (tar - self.inp_mean) / torch.sqrt(self.inp_var)

            if inp.dtype != self.dtype:
                inp = inp.to(dtype=self.dtype)
                tar = tar.to(dtype=self.dtype)

        else:
            # for now: assume only rank 0 produces valid inputs
            # a distributed model later takes care of scattering these onto
            # the participating ranks
            # to simplify things: return empty dummy tensors on other ranks
            inp = torch.empty(
                (3, self.nlat, self.nlon), device=self.device, dtype=self.dtype
            )
            tar = torch.empty(
                (3, self.nlat, self.nlon), device=self.device, dtype=self.dtype
            )
            # to show that these "dummy inputs" indeed don't play a role at all
            # multiply them with "NaN"
            inp = inp * float("nan")
            tar = tar * float("nan")

        return inp, tar
