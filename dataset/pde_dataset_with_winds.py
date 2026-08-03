import os
import torch

from dataset.shallow_water_pde_dataset import ShallowWaterPDEDataset


class PdeDatasetWithWinds(torch.utils.data.Dataset):
    """Extended dataset that computes physical winds for the PARADIS architecture.

    This dataset wraps ShallowWaterPDEDataset and computes physical winds (u, v)
    from the spectral vorticity and divergence using the getuv method. The winds are:
    1. Computed from denormalized spectral coefficients
    2. Normalized separately using wind-specific statistics derived from 100 random ICs
    """

    def __init__(
        self,
        dt,
        nsteps,
        dims=(384, 768),
        initial_condition="random",
        num_examples=32,
        device=torch.device("cpu"),
        normalize=True,
        stream=None,
        precomputed_folder=None,
        ic_kwargs=None,
        gbells_ref_ictype="random",
    ):
        self.base_dataset = ShallowWaterPDEDataset(
            dt=dt,
            nsteps=nsteps,
            dims=dims,
            device=device,
            normalize=normalize,
            stream=stream,
            precomputed_folder=precomputed_folder,
        )

        self.solver = self.base_dataset.solver
        self.nlat = self.base_dataset.nlat
        self.nlon = self.base_dataset.nlon
        self.grid = "equiangular"  # ShallowWaterSolver always uses equiangular
        self.nsteps = self.base_dataset.nsteps
        self.normalize = normalize
        self.device = device

        gbells_kwargs = ic_kwargs if initial_condition in ("gbells", "gbells_h", "gbells_h_rv") else None
        wc2_kwargs    = ic_kwargs if initial_condition == "williamson_case2" else None
        wc6_kwargs    = ic_kwargs if initial_condition in ("williamson_case6", "williamson_case6_r4") else None
        self.set_initial_condition(initial_condition, precomputed_folder=precomputed_folder,
                                   gbells_kwargs=gbells_kwargs, wc2_kwargs=wc2_kwargs, wc6_kwargs=wc6_kwargs,
                                   gbells_ref_ictype=gbells_ref_ictype)
        self.set_num_examples(num_examples)

        if self.normalize:
            self._compute_inp_statistics()
            self._compute_wind_statistics()

        self.inp_mean = self.base_dataset.inp_mean
        self.inp_var  = self.base_dataset.inp_var

    def _compute_inp_statistics(self, num_samples=20):
        """Compute mean and variance for field normalization over multiple samples."""
        inp_samples = []

        with torch.no_grad():
            for _ in range(num_samples):
                inp, _ = self.base_dataset._get_sample()
                inp_samples.append(inp)

        inp_samples = torch.stack(inp_samples, dim=0)

        self.base_dataset.inp_mean = torch.mean(inp_samples, dim=(0, 2, 3), keepdim=True).reshape(-1, 1, 1)
        self.base_dataset.inp_var  = torch.var(inp_samples,  dim=(0, 2, 3), keepdim=True).reshape(-1, 1, 1)
        self.base_dataset.inp_var  = torch.maximum(
            self.base_dataset.inp_var, torch.ones_like(self.base_dataset.inp_var) * 1e-20
        )

    def _compute_wind_statistics(self, num_samples=20):
        """Compute mean and variance for wind normalization.

        Uses 100 initial conditions rather than a single sample so that
        the statistics are stable across the range of flows seen during training.
        """
        wind_samples = []

        with torch.no_grad():
            for i in range(num_samples):
                if self.ictype == "random":
                    inp_spec = self.solver.random_initial_condition(mach=0.2)
                elif self.ictype == "galewsky":
                    inp_spec = self.solver.galewsky_initial_condition()
                elif self.ictype == "gbells":
                    ref_mean = self.base_dataset.gbells_ref_mean
                    ref_std  = self.base_dataset.gbells_ref_std
                    inp_spec = self.solver.gaussian_bells_initial_condition(
                        ref_mean, ref_std, **self.base_dataset.gbells_kwargs
                    )
                elif self.ictype == "gbells_h":
                    ref_mean = self.base_dataset.gbells_ref_mean
                    ref_std  = self.base_dataset.gbells_ref_std
                    inp_spec = self.solver.gaussian_bells_height_initial_condition(
                        ref_mean, ref_std, **self.base_dataset.gbells_kwargs
                    )
                elif self.ictype == "gbells_h_rv":
                    ref_mean = self.base_dataset.gbells_ref_mean
                    ref_std  = self.base_dataset.gbells_ref_std
                    inp_spec = self.solver.gaussian_bells_height_random_vortdiv_initial_condition(
                        ref_mean, ref_std, **self.base_dataset.gbells_kwargs
                    )
                elif self.ictype == "williamson_case2":
                    inp_spec = self.solver.williamson_case2_initial_condition(**self.base_dataset.wc2_kwargs)
                elif self.ictype == "williamson_case6":
                    inp_spec = self.solver.williamson_case6_initial_condition(**self.base_dataset.wc6_kwargs)
                elif self.ictype == "williamson_case6_standard":
                    inp_spec = self.solver.williamson_case6_initial_condition(
                        r_min=4, r_max=4, omega_min=7.848e-6, omega_max=7.848e-6, h0_min=8000.0, h0_max=8000.0,
                    )
                elif self.ictype == "williamson_case6_r4":
                    inp_spec = self.solver.williamson_case6_initial_condition(**{**self.base_dataset.wc6_kwargs, "r_min": 4, "r_max": 4})
                elif self.ictype == "precomputed":
                    inp_spec = self.solver.precomputed_initial_condition(self.precomputed_folder, i, step=0)

                winds = self.solver.getuv(inp_spec[1:])
                wind_samples.append(winds)

        wind_samples = torch.stack(wind_samples, dim=0)

        self.wind_mean = torch.mean(wind_samples, dim=(0, 2, 3), keepdim=True).reshape(
            2, 1, 1
        )
        self.wind_var = torch.var(wind_samples, dim=(0, 2, 3), keepdim=True).reshape(
            2, 1, 1
        )
        self.wind_var = torch.maximum(
            self.wind_var, torch.ones_like(self.wind_var) * 1e-8
        )

    def __len__(self):
        return len(self.base_dataset)

    def set_initial_condition(self, ictype="random", precomputed_folder=None, gbells_kwargs=None,
                              wc6_kwargs=None, wc2_kwargs=None, gbells_ref_ictype="random"):
        """Set the initial condition type."""
        self.base_dataset.set_initial_condition(
            ictype, precomputed_folder=precomputed_folder,
            gbells_kwargs=gbells_kwargs, wc6_kwargs=wc6_kwargs, wc2_kwargs=wc2_kwargs,
            gbells_ref_ictype=gbells_ref_ictype,
        )
        self.ictype = ictype
        if ictype == "precomputed":
            self.precomputed_folder = precomputed_folder

    def set_num_examples(self, num_examples=32):
        """Set the number of examples."""
        self.base_dataset.set_num_examples(num_examples)

    def _get_sample_with_winds(self, index=None):
        """Generate a sample with both fields and winds."""
        if self.ictype == "random":
            inp_spec = self.solver.random_initial_condition(mach=0.2)
            tar_spec = self.solver.timestep(inp_spec, self.nsteps)
        elif self.ictype == "galewsky":
            inp_spec = self.solver.galewsky_initial_condition()
            tar_spec = self.solver.timestep(inp_spec, self.nsteps)
        elif self.ictype == "gbells":
            ref_mean = self.base_dataset.gbells_ref_mean
            ref_std  = self.base_dataset.gbells_ref_std
            inp_spec = self.solver.gaussian_bells_initial_condition(
                ref_mean, ref_std, **self.base_dataset.gbells_kwargs
            )
            tar_spec = self.solver.timestep(inp_spec, self.nsteps)
        elif self.ictype == "gbells_h":
            ref_mean = self.base_dataset.gbells_ref_mean
            ref_std  = self.base_dataset.gbells_ref_std
            inp_spec = self.solver.gaussian_bells_height_initial_condition(
                ref_mean, ref_std, **self.base_dataset.gbells_kwargs
            )
            tar_spec = self.solver.timestep(inp_spec, self.nsteps)
        elif self.ictype == "gbells_h_rv":
            ref_mean = self.base_dataset.gbells_ref_mean
            ref_std  = self.base_dataset.gbells_ref_std
            inp_spec = self.solver.gaussian_bells_height_random_vortdiv_initial_condition(
                ref_mean, ref_std, **self.base_dataset.gbells_kwargs
            )
            tar_spec = self.solver.timestep(inp_spec, self.nsteps)
        elif self.ictype == "williamson_case2":
            inp_spec = self.solver.williamson_case2_initial_condition(**self.base_dataset.wc2_kwargs)
            tar_spec = self.solver.timestep(inp_spec, self.nsteps)
        elif self.ictype == "williamson_case6":
            inp_spec = self.solver.williamson_case6_initial_condition(**self.base_dataset.wc6_kwargs)
            tar_spec = self.solver.timestep(inp_spec, self.nsteps)
        elif self.ictype == "williamson_case6_standard":
            inp_spec = self.solver.williamson_case6_initial_condition(
                r_min=4, r_max=4, omega_min=7.848e-6, omega_max=7.848e-6, h0_min=8000.0, h0_max=8000.0,
            )
            tar_spec = self.solver.timestep(inp_spec, self.nsteps)
        elif self.ictype == "williamson_case6_r4":
            inp_spec = self.solver.williamson_case6_initial_condition(**{**self.base_dataset.wc6_kwargs, "r_min": 4, "r_max": 4})
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
            raise NotImplementedError(f"Initial Condition {self.ictype} not implemented.")

        inp_fields = self.solver.spec2grid(inp_spec)
        tar_fields = self.solver.spec2grid(tar_spec)

        inp_winds = self.solver.getuv(inp_spec[1:])
        tar_winds = self.solver.getuv(tar_spec[1:])

        return inp_fields, inp_winds, tar_fields, tar_winds

    def __getitem__(self, index):
        """Get a sample with fields and winds."""
        with torch.inference_mode():
            with torch.no_grad():
                inp_fields, inp_winds, tar_fields, tar_winds = (
                    self._get_sample_with_winds(index=index)
                )

                if self.normalize:
                    inp_fields = (inp_fields - self.inp_mean) / torch.sqrt(self.inp_var)
                    tar_fields = (tar_fields - self.inp_mean) / torch.sqrt(self.inp_var)

                    inp_winds = (inp_winds - self.wind_mean) / torch.sqrt(self.wind_var)
                    tar_winds = (tar_winds - self.wind_mean) / torch.sqrt(self.wind_var)

        return (
            inp_fields.clone(),
            inp_winds.clone(),
            tar_fields.clone(),
            tar_winds.clone(),
        )
