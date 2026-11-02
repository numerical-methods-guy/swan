import torch

from kdv.tkdv_dataset import TKdVPDEDataset


class TKdVDatasetWithForcing(torch.utils.data.Dataset):
    """
    Wrapper around TKdVPDEDataset that optionally computes an advection velocity
    channel derived from the physical field u.

    In tKdV the nonlinear term is u·u_x, so u itself acts as the advection velocity.
    When use_advection_vel=True, the dataset returns
        (inp, inp_vel, tar, tar_vel)
    where inp_vel = inp (u is its own advection velocity).
    When use_advection_vel=False, returns
        (inp, tar)
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
        use_advection_vel: bool = False,
    ):
        self.use_advection_vel = use_advection_vel

        self.base_dataset = TKdVPDEDataset(
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
        )

        self.solver = self.base_dataset.solver
        self.normalize = normalize
        self.device = device
        self.ictype = self.base_dataset.ictype

        if self.normalize:
            self._compute_inp_statistics()
            if self.use_advection_vel:
                self._compute_vel_statistics()

        self.inp_mean = self.base_dataset.inp_mean
        self.inp_var = self.base_dataset.inp_var

    def _compute_inp_statistics(self, num_samples: int = 20):
        inp_samples = []
        with torch.no_grad():
            for i in range(num_samples):
                idx = i if self.ictype == "precomputed" else None
                inp, _ = self.base_dataset._get_sample(index=idx)
                inp_samples.append(inp)
        inp_samples = torch.stack(inp_samples, dim=0)
        self.base_dataset.inp_mean = inp_samples.mean()
        self.base_dataset.inp_var = inp_samples.var().clamp(min=1e-20)

    def _compute_vel_statistics(self, num_samples: int = 20):
        vel_samples = []
        with torch.no_grad():
            for i in range(num_samples):
                idx = i if self.ictype == "precomputed" else None
                inp, _ = self.base_dataset._get_sample(index=idx)
                vel_samples.append(self._advection_vel(inp))
        vel_samples = torch.stack(vel_samples, dim=0)
        self.vel_mean = vel_samples.mean()
        self.vel_var = vel_samples.var().clamp(min=1e-20)

    def _advection_vel(self, u: torch.Tensor) -> torch.Tensor:
        """Return the advection velocity channel from physical field u. Shape: same as u."""
        return u

    def __len__(self):
        return len(self.base_dataset)

    def set_initial_condition(self, ictype: str = "gibbs", precomputed_folder: str = None):
        self.base_dataset.set_initial_condition(ictype, precomputed_folder)
        self.ictype = ictype

    def set_num_examples(self, num_examples: int = 32):
        self.base_dataset.set_num_examples(num_examples)

    def __getitem__(self, index: int):
        with torch.inference_mode():
            with torch.no_grad():
                inp_raw, tar_raw = self.base_dataset._get_sample(index=index)

                if self.normalize:
                    inp_std = torch.sqrt(self.base_dataset.inp_var)
                    inp = (inp_raw - self.base_dataset.inp_mean) / inp_std
                    tar = (tar_raw - self.base_dataset.inp_mean) / inp_std
                else:
                    inp, tar = inp_raw, tar_raw

                if not self.use_advection_vel:
                    return inp.clone(), tar.clone()

                inp_vel = self._advection_vel(inp_raw)
                tar_vel = self._advection_vel(tar_raw)

                if self.normalize:
                    vel_std = torch.sqrt(self.vel_var)
                    inp_vel = (inp_vel - self.vel_mean) / vel_std
                    tar_vel = (tar_vel - self.vel_mean) / vel_std

        return inp.clone(), inp_vel.clone(), tar.clone(), tar_vel.clone()
