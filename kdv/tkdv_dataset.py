import os
import torch

from kdv.tkdv_solver import TKdVSolver


class TKdVPDEDataset(torch.utils.data.Dataset):
    """
    Dataset class for the truncated KdV equation.

    Each sample is an (input, target) pair of physical fields on the J-point grid,
    where target is the state after nsteps solver timesteps from the input.
    """

    def __init__(
        self,
        dt: float,
        nsteps: int,
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
    ):
        self.dt = dt
        self.nsteps = nsteps
        self.dt_solver = dt / float(nsteps)
        self.J = J
        self.theta = theta
        self.num_examples = num_examples
        self.device = device
        self.normalize = normalize
        self.precomputed_folder = precomputed_folder
        self.gibbs_burnin = gibbs_burnin
        self.gibbs_thin = gibbs_thin

        self.solver = TKdVSolver(J=J, dt=self.dt_solver, D=D, E0=E0, L0=L0).to(self.device)

        self.set_initial_condition(ictype=initial_condition)

        if self.normalize:
            idx = 0 if initial_condition == "precomputed" else None
            with torch.no_grad():
                inp0, _ = self._get_sample(index=idx)
            self.inp_mean = inp0.mean()
            self.inp_var = inp0.var().clamp(min=1e-20)

    def _compute_inp_statistics(self, num_samples: int = 20):
        inp_samples = []
        with torch.no_grad():
            for i in range(num_samples):
                idx = i if self.ictype == "precomputed" else None
                inp, _ = self._get_sample(index=idx)
                inp_samples.append(inp)
        inp_samples = torch.stack(inp_samples, dim=0)
        self.inp_mean = inp_samples.mean()
        self.inp_var = inp_samples.var().clamp(min=1e-20)

    def __len__(self):
        return self.num_examples

    def set_num_examples(self, num_examples: int = 32):
        self.num_examples = num_examples

    def set_initial_condition(self, ictype: str = "gibbs", precomputed_folder: str = None):
        self.ictype = ictype
        if ictype == "precomputed":
            if precomputed_folder is None and self.precomputed_folder is None:
                raise ValueError("precomputed_folder must be provided when ictype='precomputed'")
            if precomputed_folder is not None:
                self.precomputed_folder = precomputed_folder
        elif ictype == "gibbs":
            pass
        else:
            raise ValueError(f"Unknown ictype: {ictype!r}. Choose 'gibbs' or 'precomputed'.")

    def _get_sample(self, index: int = None):
        with torch.no_grad():
            if self.ictype == "gibbs":
                inp_hat = self.solver.gibbs_initial_condition(
                    theta=self.theta,
                    n_burnin=self.gibbs_burnin,
                    n_samples=1,
                    thin=self.gibbs_thin,
                )
            elif self.ictype == "precomputed":
                if index is None:
                    raise ValueError("index must be provided when ictype='precomputed'")
                inp_hat = self.solver.precomputed_initial_condition(
                    self.precomputed_folder, index, step=0
                )
                target_path = os.path.join(self.precomputed_folder, f"{index}_1.pt")
                if os.path.exists(target_path):
                    tar_hat = self.solver.precomputed_initial_condition(
                        self.precomputed_folder, index, step=1
                    )
                    inp = self.solver.spec2phys(inp_hat).float()
                    tar = self.solver.spec2phys(tar_hat).float()
                    return inp, tar

            tar_hat = self.solver.timestep(inp_hat, nsteps=self.nsteps)

            inp = self.solver.spec2phys(inp_hat).float()
            tar = self.solver.spec2phys(tar_hat).float()

        return inp, tar

    def __getitem__(self, index: int):
        with torch.inference_mode():
            with torch.no_grad():
                inp, tar = self._get_sample(index=index)

                if self.normalize:
                    inp_std = torch.sqrt(self.inp_var)
                    inp = (inp - self.inp_mean) / inp_std
                    tar = (tar - self.inp_mean) / inp_std

        return inp.clone(), tar.clone()
