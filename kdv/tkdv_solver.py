import numpy as np
import torch
import torch.nn as nn


class TKdVSolver(nn.Module):
    """
    Solver for the truncated KdV (tKdV) equation on a periodic domain x in [-pi, pi].

    The equation (nondimensional, Majda-Moore-Qi 2019) is:
        u_t + D^{-3/2} E0^{1/2} L0^{-3/2} * d/dx P_Lambda(u^2/2)
             + D^{1/2} L0^{-3} * u_xxx = 0

    State is represented as spectral coefficients u_hat[k] for k = 1..Lambda
    (zero mean enforced: u_hat[0] = 0).
    Energy is normalized to 1: 2*pi * sum_{k=1}^{Lambda} |u_hat[k]|^2 = 1.
    """

    def __init__(
        self,
        J: int = 32,
        dt: float = 0.01,
        D: float = 1.0,
        E0: float = 100.0,
        L0: float = 6.0,
    ):
        super().__init__()

        assert J % 2 == 0, "J must be even"
        self.J = J
        self.Lambda = J // 2
        self.dt = dt

        # physical grid x in [-pi, pi), shape (J,)
        x = torch.linspace(-torch.pi, torch.pi, J + 1)[:-1].double()
        self.register_buffer("x", x)

        # wavenumbers k = 1..Lambda, shape (Lambda,)
        k = torch.arange(1, self.Lambda + 1).double()
        self.register_buffer("k", k)

        # Hamiltonian coefficients: H = h3*H3 - h2*H2
        # H3 = (1/6) int u^3 dx,  H2 = (1/2) int (u_x)^2 dx = 2pi * sum_k k^2 |u_hat_k|^2
        # h3 = h3_pde, h2 = h2_pde (no extra pi; supp eq [14] uses a different H3 normalization)
        h3 = torch.tensor(
            E0 ** 0.5 * L0 ** (-1.5) * D ** (-1.5), dtype=torch.float64
        )
        h2 = torch.tensor(L0 ** (-3) * D ** 0.5, dtype=torch.float64)
        self.register_buffer("h3", h3)
        self.register_buffer("h2", h2)

        # RHS coefficients (from the PDE written in spectral space)
        # nonlinear:  -i*k * h3_pde * (u^2/2)_hat[k]   where h3_pde = D^{-3/2} E0^{1/2} L0^{-3/2}
        # dispersion: -i*k^3 * h2_pde * u_hat[k]         where h2_pde = D^{1/2} L0^{-3}
        h3_pde = torch.tensor(
            D ** (-1.5) * E0 ** 0.5 * L0 ** (-1.5), dtype=torch.float64
        )
        h2_pde = torch.tensor(D ** 0.5 * L0 ** (-3), dtype=torch.float64)
        self.register_buffer("h3_pde", h3_pde)
        self.register_buffer("h2_pde", h2_pde)

    def phys2spec(self, u: torch.Tensor) -> torch.Tensor:
        """
        Physical grid → truncated spectral coefficients.

        Args:
            u: real field on J grid points, shape (..., J)

        Returns:
            u_hat: complex coefficients for k=1..Lambda, shape (..., Lambda)
                   normalised so that 2*pi * sum |u_hat[k]|^2 = (1/J^2) * ||u||^2_discrete
        """
        # rfft gives coefficients for k=0..J//2, with pytorch's 1/1 normalisation
        # divide by J to get the standard DFT convention: u_hat[k] = (1/J) sum_j u_j e^{-ikx_j}
        u_hat_full = torch.fft.rfft(u, dim=-1) / self.J
        # drop k=0 (zero mean) and k>Lambda (truncation), keep k=1..Lambda
        return u_hat_full[..., 1:self.Lambda + 1]

    def spec2phys(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Truncated spectral coefficients → physical grid.

        Args:
            u_hat: complex coefficients for k=1..Lambda, shape (..., Lambda)

        Returns:
            u: real field on J grid points, shape (..., J)
        """
        # pad back to full rfft size: k=0 (zero), k=1..Lambda, k>Lambda (zero)
        shape = list(u_hat.shape)
        shape[-1] = self.J // 2 + 1
        u_hat_full = torch.zeros(shape, dtype=u_hat.dtype, device=u_hat.device)
        u_hat_full[..., 1:self.Lambda + 1] = u_hat
        # irfft expects unnormalised coefficients, so multiply back by J
        return torch.fft.irfft(u_hat_full * self.J, n=self.J, dim=-1)

    def _quadratic_dealiased(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral coefficients of P_Lambda(u^2/2) using 3/2-rule dealiasing.

        Pads to J_pad = 3*Lambda grid points before squaring to prevent aliasing
        of the quadratic nonlinearity back into the resolved modes k=1..Lambda.
        """
        J_pad = 3 * self.Lambda  # dealiased grid size
        # Build padded rfft buffer: k=0 zero, k=1..Lambda from u_hat, rest zero
        shape = list(u_hat.shape[:-1]) + [J_pad // 2 + 1]
        buf = torch.zeros(shape, dtype=u_hat.dtype, device=u_hat.device)
        buf[..., 1:self.Lambda + 1] = u_hat
        # irfft(buf * J_pad) = sum_k u_hat[k] exp(2pi*i*k*j/J_pad): physical values on J_pad pts
        u_phys = torch.fft.irfft(buf * J_pad, n=J_pad, dim=-1)
        # square, transform back, truncate to k=1..Lambda
        u2_hat_full = torch.fft.rfft(u_phys ** 2 / 2, dim=-1) / J_pad
        return u2_hat_full[..., 1:self.Lambda + 1]

    def dudt(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        RHS of the tKdV equation in spectral space.

        Args:
            u_hat: complex spectral coefficients k=1..Lambda, shape (..., Lambda)

        Returns:
            du_hat/dt: same shape as u_hat
        """
        # nonlinear term: -ik * h3_pde * P_Lambda(u^2/2)_hat[k]  (dealiased)
        u2_hat = self._quadratic_dealiased(u_hat)
        nonlinear = -1j * self.k * self.h3_pde * u2_hat

        # linear dispersion term: +ik^3 * h2_pde * u_hat[k]  (since (ik)^3 = -ik^3, negating gives +ik^3)
        dispersion = +1j * self.k ** 3 * self.h2_pde * u_hat

        return nonlinear + dispersion

    def _solve_midpoint_stage(
        self,
        a: torch.Tensor,
        w: float,
        n_iter: int = 20,
    ) -> torch.Tensor:
        """
        Solve one implicit midpoint stage:
            y = a + w * dt * F[½(y + a)]
        via Picard iteration starting from y = a.
        """
        y = a.clone()
        for _ in range(n_iter):
            y = a + w * self.dt * self.dudt(0.5 * (y + a))
        return y

    def timestep(self, u_hat: torch.Tensor, nsteps: int = 1) -> torch.Tensor:
        """
        Advance u_hat forward by nsteps timesteps using the 4th-order
        implicit midpoint symplectic scheme (McLachlan 1993, SI Section C).

        Three-stage scheme:
            y1      = uⁿ + w1 * dt * F[½(y1 + uⁿ)]
            y2      = y1 + w2 * dt * F[½(y2 + y1)]
            uⁿ⁺¹   = y2 + w3 * dt * F[½(uⁿ⁺¹ + y2)]

        with w1 = w3 = (2 + 2^{1/3} + 2^{-1/3}) / 3,  w2 = 1 - 2*w1.

        Args:
            u_hat:  complex spectral coefficients, shape (..., Lambda)
            nsteps: number of timesteps to take

        Returns:
            u_hat at time t + nsteps * dt, same shape
        """
        w1 = (2.0 + 2.0 ** (1.0 / 3.0) + 2.0 ** (-1.0 / 3.0)) / 3.0
        w2 = 1.0 - 2.0 * w1
        w3 = w1

        for _ in range(nsteps):
            y1 = self._solve_midpoint_stage(u_hat, w1)
            y2 = self._solve_midpoint_stage(y1, w2)
            u_hat = self._solve_midpoint_stage(y2, w3)

        return u_hat

    def hamiltonian(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute the tKdV Hamiltonian H_Lambda = h3*H3(u) - h2*H2(u).

        H3(u) = (1/6) int u^3 dx  -- cubic term, computed in physical space
        H2(u) = (1/2) int (u_x)^2 dx = pi * sum_k k^2 |u_hat[k]|^2  -- wave slope energy

        Args:
            u_hat: complex spectral coefficients k=1..Lambda, shape (..., Lambda)

        Returns:
            scalar Hamiltonian value, shape (...)
        """
        # H3: go to physical space, integrate u^3 over [-pi, pi]
        u = self.spec2phys(u_hat)
        dx = 2.0 * torch.pi / self.J
        H3 = (1.0 / 6.0) * (u ** 3).sum(dim=-1) * dx

        # H2: wave slope energy, cheapest in spectral space
        # sum over |k|=1..Lambda (both signs): 2 * pi * sum_{k=1}^Lambda k^2 |u_hat[k]|^2
        H2 = 2.0 * torch.pi * (self.k ** 2 * u_hat.abs() ** 2).sum(dim=-1).real

        return self.h3 * H3 - self.h2 * H2

    def energy(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute the L2 energy E(u) = 2*pi * sum_{k=1}^{Lambda} |u_hat[k]|^2.

        Should remain = 1 throughout the simulation (conserved quantity).

        Args:
            u_hat: complex spectral coefficients k=1..Lambda, shape (..., Lambda)

        Returns:
            scalar energy value, shape (...)
        """
        return 2.0 * torch.pi * (u_hat.abs() ** 2).sum(dim=-1).real

    def gibbs_initial_condition(
        self,
        theta: float,
        n_burnin: int = 1000,
        n_samples: int = 1,
        thin: int = 10,
    ) -> torch.Tensor:
        """
        Sample u_hat from the Gibbs measure G_theta via Metropolis-Hastings MCMC
        on the energy sphere (SI Algorithm 1).

        Args:
            theta:     inverse temperature (negative for physical regime)
            n_burnin:  number of steps to discard before collecting samples
            n_samples: number of independent samples to return
            thin:      collect one sample every `thin` steps after burnin

        Returns:
            u_hat: shape (n_samples, Lambda) if n_samples > 1, else (Lambda,)
        """
        device = self.k.device

        # initialise on the energy sphere: Gaussian -> normalize to E=1
        u_hat = torch.randn(self.Lambda, dtype=torch.complex128, device=device)
        u_hat = u_hat / torch.sqrt(self.energy(u_hat))

        H_cur = self.hamiltonian(u_hat)

        samples = []
        n_total = n_burnin + n_samples * thin

        for step in range(n_total):
            # propose: perturb all modes with Gaussian noise scaled by Lambda
            noise = torch.randn(self.Lambda, dtype=torch.complex128, device=device)
            w_hat = u_hat + noise / self.Lambda ** 0.5
            # normalize proposal to energy sphere
            w_hat = w_hat / torch.sqrt(self.energy(w_hat))

            H_prop = self.hamiltonian(w_hat)

            # Metropolis accept/reject (proposal is symmetric after normalisation)
            log_alpha = -theta * (H_prop - H_cur)
            if torch.log(torch.rand(1, device=device)) < log_alpha:
                u_hat = w_hat
                H_cur = H_prop

            # collect sample after burnin
            if step >= n_burnin and (step - n_burnin) % thin == 0:
                samples.append(u_hat.clone())

        samples = torch.stack(samples, dim=0)  # (n_samples, Lambda)
        return samples.squeeze(0) if n_samples == 1 else samples

    def precomputed_initial_condition(
        self, folder: str, index: int, step: int = 0
    ) -> torch.Tensor:
        """
        Load a precomputed spectral state from disk.

        Expects a file named {folder}/{index}_{step}.pt containing a u_hat tensor
        of shape (Lambda,).

        Args:
            folder: directory containing the saved tensors
            index:  IC index
            step:   rollout step index (default 0 = initial condition)

        Returns:
            u_hat: complex spectral coefficients, shape (Lambda,)
        """
        import os
        path = os.path.join(folder, f"{index}_{step}.pt")
        return torch.load(path, map_location=self.k.device)
