import os
import torch

from dataset.pde_dataset_with_winds import PdeDatasetWithWinds


class MultiStepPdeDatasetWithWinds(PdeDatasetWithWinds):
    """Multi-step extension of PdeDatasetWithWinds.

    Runs the solver for input_step_idx steps to reach the NN input state,
    then collects n_rollout_steps further steps as targets.

    __getitem__ returns:
        inp_fields:  (channels, nlat, nlon)
        inp_winds:   (2, nlat, nlon)
        tar_fields:  (n_rollout_steps, channels, nlat, nlon)
        tar_winds:   (n_rollout_steps, 2, nlat, nlon)
    """

    def __init__(
        self,
        dt,
        nsteps,
        n_rollout_steps,
        input_step_idx=0,
        dims=(384, 768),
        initial_condition="random",
        num_examples=32,
        device=torch.device("cpu"),
        normalize=True,
        stream=None,
        precomputed_folder=None,
        ic_kwargs=None,
    ):
        if n_rollout_steps < 1:
            raise ValueError(f"n_rollout_steps must be at least 1, got {n_rollout_steps}")
        if input_step_idx < 0:
            raise ValueError(f"input_step_idx must be non-negative, got {input_step_idx}")

        self.n_rollout_steps = n_rollout_steps
        self.input_step_idx = input_step_idx

        super().__init__(
            dt=dt,
            nsteps=nsteps,
            dims=dims,
            initial_condition=initial_condition,
            num_examples=num_examples,
            device=device,
            normalize=normalize,
            stream=stream,
            precomputed_folder=precomputed_folder,
            ic_kwargs=ic_kwargs,
        )

    def _compute_inp_statistics(self, num_samples=20):
        """Compute field stats over all rollout steps and the input step."""
        inp_samples = []

        with torch.no_grad():
            for i in range(num_samples):
                inp_fields, _, tar_fields, _ = self._get_sample_with_winds(index=i)
                inp_samples.append(inp_fields)
                for s in range(self.n_rollout_steps):
                    inp_samples.append(tar_fields[s])

        inp_samples = torch.stack(inp_samples, dim=0)

        self.base_dataset.inp_mean = torch.mean(inp_samples, dim=(0, 2, 3), keepdim=True).reshape(-1, 1, 1)
        self.base_dataset.inp_var  = torch.var(inp_samples,  dim=(0, 2, 3), keepdim=True).reshape(-1, 1, 1)
        self.base_dataset.inp_var  = torch.maximum(
            self.base_dataset.inp_var, torch.ones_like(self.base_dataset.inp_var) * 1e-8
        )

    def _compute_wind_statistics(self, num_samples=20):
        """Compute wind stats over all rollout steps and the input step."""
        wind_samples = []

        with torch.no_grad():
            for i in range(num_samples):
                _, inp_winds, _, tar_winds = self._get_sample_with_winds(index=i)
                wind_samples.append(inp_winds)
                for s in range(self.n_rollout_steps):
                    wind_samples.append(tar_winds[s])

        wind_samples = torch.stack(wind_samples, dim=0)

        self.wind_mean = torch.mean(wind_samples, dim=(0, 2, 3), keepdim=True).reshape(2, 1, 1)
        self.wind_var  = torch.var(wind_samples,  dim=(0, 2, 3), keepdim=True).reshape(2, 1, 1)
        self.wind_var  = torch.maximum(
            self.wind_var, torch.ones_like(self.wind_var) * 1e-8
        )

    def _get_sample_precomputed(self, index):
        """Load a precomputed trajectory, falling back to solver.timestep for missing files."""
        # step 0 must always exist
        current_spec = self.solver.precomputed_initial_condition(self.precomputed_folder, index, step=0)

        # advance to input_step_idx, loading precomputed files where available
        for s in range(self.input_step_idx):
            path = os.path.join(self.precomputed_folder, f"{index}_{s + 1}.pt")
            if os.path.exists(path):
                current_spec = self.solver.precomputed_initial_condition(self.precomputed_folder, index, step=s + 1)
            else:
                current_spec = self.solver.timestep(current_spec, self.nsteps)

        inp_fields = self.solver.spec2grid(current_spec)
        inp_winds  = self.solver.getuv(current_spec[1:])

        tar_fields_list = []
        tar_winds_list  = []

        for k in range(1, self.n_rollout_steps + 1):
            path = os.path.join(self.precomputed_folder, f"{index}_{self.input_step_idx + k}.pt")
            if os.path.exists(path):
                current_spec = self.solver.precomputed_initial_condition(self.precomputed_folder, index, step=self.input_step_idx + k)
            else:
                current_spec = self.solver.timestep(current_spec, self.nsteps)
            tar_fields_list.append(self.solver.spec2grid(current_spec))
            tar_winds_list.append(self.solver.getuv(current_spec[1:]))

        tar_fields = torch.stack(tar_fields_list, dim=0)
        tar_winds  = torch.stack(tar_winds_list,  dim=0)

        return inp_fields, inp_winds, tar_fields, tar_winds

    def _get_sample_with_winds(self, index=None):
        """Generate input at input_step_idx and n_rollout_steps targets beyond it."""
        if self.ictype == "precomputed":
            if index is None:
                raise ValueError("index must be provided when ictype='precomputed'")
            return self._get_sample_precomputed(index)

        if self.ictype == "random":
            spec = self.solver.random_initial_condition(mach=0.2)
        elif self.ictype == "galewsky":
            spec = self.solver.galewsky_initial_condition()
        elif self.ictype == "gbells":
            ref_mean = self.base_dataset.gbells_ref_mean
            ref_std  = self.base_dataset.gbells_ref_std
            spec = self.solver.gaussian_bells_initial_condition(
                ref_mean, ref_std, **self.base_dataset.gbells_kwargs
            )
        elif self.ictype == "gbells_h":
            ref_mean = self.base_dataset.gbells_ref_mean
            ref_std  = self.base_dataset.gbells_ref_std
            spec = self.solver.gaussian_bells_height_initial_condition(
                ref_mean, ref_std, **self.base_dataset.gbells_kwargs
            )
        elif self.ictype == "williamson_case2":
            spec = self.solver.williamson_case2_initial_condition(**self.base_dataset.wc2_kwargs)
        elif self.ictype == "williamson_case6":
            spec = self.solver.williamson_case6_initial_condition(**self.base_dataset.wc6_kwargs)
        else:
            raise NotImplementedError(f"Initial Condition {self.ictype} not implemented.")

        # advance to input_step_idx — discard intermediate states
        if self.input_step_idx > 0:
            spec = self.solver.timestep(spec, self.nsteps * self.input_step_idx)

        inp_fields = self.solver.spec2grid(spec)
        inp_winds  = self.solver.getuv(spec[1:])

        tar_fields_list = []
        tar_winds_list  = []

        for _ in range(self.n_rollout_steps):
            spec = self.solver.timestep(spec, self.nsteps)
            tar_fields_list.append(self.solver.spec2grid(spec))
            tar_winds_list.append(self.solver.getuv(spec[1:]))

        tar_fields = torch.stack(tar_fields_list, dim=0)  # (n_rollout_steps, channels, nlat, nlon)
        tar_winds  = torch.stack(tar_winds_list,  dim=0)  # (n_rollout_steps, 2, nlat, nlon)

        return inp_fields, inp_winds, tar_fields, tar_winds

    def __getitem__(self, index):
        """Get a multi-step sample with fields and winds."""
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
