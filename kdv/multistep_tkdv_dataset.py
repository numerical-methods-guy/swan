import os
import torch

from kdv.tkdv_dataset_with_forcing import TKdVDatasetWithForcing


class MultiStepTKdVDataset(TKdVDatasetWithForcing):
    """Multi-step extension of TKdVDatasetWithForcing.

    Runs the solver for input_step_idx steps to reach the NN input state,
    then collects n_rollout_steps further steps as targets.

    When use_advection_vel=False, __getitem__ returns:
        inp:      (J,)
        tar:      (n_rollout_steps, J)

    When use_advection_vel=True, __getitem__ returns:
        inp:      (J,)
        inp_vel:  (J,)
        tar:      (n_rollout_steps, J)
        tar_vel:  (n_rollout_steps, J)
    """

    def __init__(
        self,
        dt: float,
        nsteps: int,
        n_rollout_steps: int,
        input_step_idx: int = 0,
        J: int = 32,
        D: float = 1.0,
        E0: float = 100.0,
        L0: float = 6.0,
        theta: float = -0.1,
        initial_condition: str = "gibbs",
        num_examples: int = 32,
        device: torch.device = torch.device("cpu"),
        normalize: bool = True,
        precomputed_folder: str = None,
        gibbs_burnin: int = 1000,
        gibbs_thin: int = 10,
        use_advection_vel: bool = False,
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
            J=J,
            D=D,
            E0=E0,
            L0=L0,
            theta=theta,
            initial_condition=initial_condition,
            num_examples=num_examples,
            device=device,
            normalize=normalize,
            precomputed_folder=precomputed_folder,
            gibbs_burnin=gibbs_burnin,
            gibbs_thin=gibbs_thin,
            use_advection_vel=use_advection_vel,
        )

    def _compute_vel_statistics(self, num_samples: int = 20):
        """Compute vel stats over input step and all rollout target steps."""
        vel_samples = []
        with torch.no_grad():
            for i in range(num_samples):
                inp, tar = self._get_sample_multistep(index=i, normalize=False)
                vel_samples.append(self._advection_vel(inp))
                for s in range(self.n_rollout_steps):
                    vel_samples.append(self._advection_vel(tar[s]))

        vel_samples = torch.stack(vel_samples, dim=0)
        self.vel_mean = vel_samples.mean()
        self.vel_var = vel_samples.var().clamp(min=1e-20)

    def _compute_inp_statistics(self, num_samples: int = 20):
        """Compute field stats over input step and all rollout target steps."""
        inp_samples = []
        with torch.no_grad():
            for i in range(num_samples):
                inp, tar = self._get_sample_multistep(index=i, normalize=False)
                inp_samples.append(inp)
                for s in range(self.n_rollout_steps):
                    inp_samples.append(tar[s])

        inp_samples = torch.stack(inp_samples, dim=0)
        self.base_dataset.inp_mean = inp_samples.mean()
        self.base_dataset.inp_var = inp_samples.var().clamp(min=1e-20)

    def _get_sample_precomputed(self, index: int):
        """Load precomputed trajectory, falling back to solver.timestep for missing steps."""
        solver = self.solver

        current_hat = solver.precomputed_initial_condition(
            self.base_dataset.precomputed_folder, index, step=0
        )

        for s in range(self.input_step_idx):
            path = os.path.join(
                self.base_dataset.precomputed_folder, f"{index}_{s + 1}.pt"
            )
            if os.path.exists(path):
                current_hat = solver.precomputed_initial_condition(
                    self.base_dataset.precomputed_folder, index, step=s + 1
                )
            else:
                current_hat = solver.timestep(current_hat, self.base_dataset.nsteps)

        inp = solver.spec2phys(current_hat).float()

        tar_list = []
        for k in range(1, self.n_rollout_steps + 1):
            path = os.path.join(
                self.base_dataset.precomputed_folder,
                f"{index}_{self.input_step_idx + k}.pt",
            )
            if os.path.exists(path):
                current_hat = solver.precomputed_initial_condition(
                    self.base_dataset.precomputed_folder,
                    index,
                    step=self.input_step_idx + k,
                )
            else:
                current_hat = solver.timestep(current_hat, self.base_dataset.nsteps)
            tar_list.append(solver.spec2phys(current_hat).float())

        tar = torch.stack(tar_list, dim=0)  # (n_rollout_steps, J)
        return inp, tar

    def _get_sample_multistep(self, index: int = None, normalize: bool = False):
        """Generate input at input_step_idx and n_rollout_steps targets beyond it."""
        solver = self.solver
        ictype = self.base_dataset.ictype

        if ictype == "precomputed":
            if index is None:
                raise ValueError("index must be provided when ictype='precomputed'")
            inp, tar = self._get_sample_precomputed(index)
        elif ictype == "gibbs":
            inp_hat = solver.gibbs_initial_condition(
                theta=self.base_dataset.theta,
                n_burnin=self.base_dataset.gibbs_burnin,
                n_samples=1,
                thin=self.base_dataset.gibbs_thin,
            )
            if self.input_step_idx > 0:
                inp_hat = solver.timestep(inp_hat, self.base_dataset.nsteps * self.input_step_idx)
            inp = solver.spec2phys(inp_hat).float()

            tar_list = []
            for _ in range(self.n_rollout_steps):
                inp_hat = solver.timestep(inp_hat, self.base_dataset.nsteps)
                tar_list.append(solver.spec2phys(inp_hat).float())
            tar = torch.stack(tar_list, dim=0)
        else:
            raise NotImplementedError(f"ictype {ictype!r} not implemented in multistep sampler")

        if normalize:
            dev = inp.device
            inp_mean = self.base_dataset.inp_mean.to(dev)
            inp_std = torch.sqrt(self.base_dataset.inp_var.to(dev))
            inp = (inp - inp_mean) / inp_std
            tar = (tar - inp_mean) / inp_std

        return inp, tar

    def __getitem__(self, index: int):
        with torch.inference_mode():
            with torch.no_grad():
                inp, tar = self._get_sample_multistep(
                    index=index,
                    normalize=self.normalize,
                )

                if not self.use_advection_vel:
                    return inp.cpu().clone(), tar.cpu().clone()

                if self.normalize:
                    dev = inp.device
                    inp_std = torch.sqrt(self.base_dataset.inp_var.to(dev))
                    inp_mean = self.base_dataset.inp_mean.to(dev)
                    inp_raw = inp * inp_std + inp_mean
                    tar_raw = tar * inp_std + inp_mean
                else:
                    inp_raw, tar_raw = inp, tar

                inp_vel = self._advection_vel(inp_raw)
                tar_vel = torch.stack([
                    self._advection_vel(tar_raw[s])
                    for s in range(self.n_rollout_steps)
                ], dim=0)

                if self.normalize:
                    dev = inp_vel.device
                    vel_std = torch.sqrt(self.vel_var.to(dev))
                    vel_mean = self.vel_mean.to(dev)
                    inp_vel = (inp_vel - vel_mean) / vel_std
                    tar_vel = (tar_vel - vel_mean) / vel_std

        return inp.cpu().clone(), inp_vel.cpu().clone(), tar.cpu().clone(), tar_vel.cpu().clone()
