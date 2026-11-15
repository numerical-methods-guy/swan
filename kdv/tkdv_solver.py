import numpy as np
import torch
import torch.nn as nn


class TKdVSolver(nn.Module):
    """
    Solver for the truncated KdV (tKdV) equation on a periodic domain x in [-L0*pi/2, L0*pi/2).

    The equation (nondimensional, Majda-Moore-Qi 2019) is:
        u_t + D^{-3/2} E0^{1/2} L0^{-3/2} * d/dx P_Lambda(u^2/2)
             + D^{1/2} L0^{-3} * u_xxx = 0

    State is represented as full spectral coefficients u_hat[k] for k = 0..J-1
    in FFT order: [0, 1, ..., J/2, -J/2+1, ..., -1].
    Zero mean enforced: u_hat[0] = 0.  Nyquist zeroed: u_hat[J//2] = 0.
    Energy is normalized to 1: 2*pi * 0.5 * sum_{k!=0} |u_hat[k]|^2 = 1.
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
        self.Lambda = J // 2  # kept for reference; state size is now J
        self.dt = dt

        # physical grid x in [-L0*pi/2, L0*pi/2), shape (J,)
        x = torch.linspace(-L0 / 2 * torch.pi, L0 / 2 * torch.pi, J + 1)[:-1].double()
        self.register_buffer("x", x)

        # wavenumbers in FFT order: [0, 1, ..., J/2, -J/2+1, ..., -1], shape (J,)
        k = torch.cat([torch.arange(0, J // 2 + 1), torch.arange(-J // 2 + 1, 0)]).double()
        self.register_buffer("k", k)

        # Hamiltonian coefficients
        h3 = torch.tensor(E0 ** 0.5 * L0 ** (-1.5) * D ** (-1.5), dtype=torch.float64)
        h2 = torch.tensor(L0 ** (-3) * D ** 0.5, dtype=torch.float64)
        self.register_buffer("h3", h3)
        self.register_buffer("h2", h2)

        # PDE RHS coefficients
        h3_pde = torch.tensor(D ** (-1.5) * E0 ** 0.5 * L0 ** (-1.5), dtype=torch.float64)
        h2_pde = torch.tensor(D ** 0.5 * L0 ** (-3), dtype=torch.float64)
        self.register_buffer("h3_pde", h3_pde)
        self.register_buffer("h2_pde", h2_pde)

    def phys2spec(self, u: torch.Tensor) -> torch.Tensor:
        """
        Physical grid → full spectral coefficients (FFT order, length J).

        Zeroes k=0 (mean) and k=J//2 (Nyquist).
        Convention: u_hat[k] = (1/J) sum_j u_j exp(-2*pi*i*k*j/J)
        so that 2*pi * 0.5 * sum_{k!=0} |u_hat[k]|^2 is the L2 energy.
        """
        u_hat = torch.fft.fft(u.double(), dim=-1) / self.J
        u_hat[..., 0] = 0.0
        u_hat[..., self.J // 2] = 0.0
        return u_hat

    def spec2phys(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Full spectral coefficients (FFT order, length J) → physical grid.
        """
        return torch.fft.ifft(u_hat * self.J, dim=-1).real

    def _quadratic_dealiased(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute (u^2)_hat using 3/2-rule dealiasing (matches MATLAB u2k_dealiasing.m).

        Pads to J_pad = 3*J//2 grid points before squaring.
        Returns full J-length spectrum of u^2 (NOT u^2/2).
        Zeros at k=0 and k=J//2.
        """
        J_pad = 3 * self.J // 2  # = 48 for J=32

        shape = list(u_hat.shape[:-1]) + [J_pad]
        U_hat = torch.zeros(shape, dtype=u_hat.dtype, device=u_hat.device)

        # positive freqs k=0..J/2-1 → indices 0..J/2-1 of J_pad array
        U_hat[..., :self.J // 2] = 1.5 * u_hat[..., :self.J // 2]
        # negative freqs k=-J/2+1..-1 → indices J+1..3J/2-1 of J_pad array
        U_hat[..., self.J + 1:J_pad] = 1.5 * u_hat[..., self.J // 2 + 1:self.J]

        u_phys = torch.fft.ifft(U_hat, dim=-1).real
        u2_phys = u_phys ** 2

        U2_hat = (2.0 / 3.0) * torch.fft.fft(u2_phys, dim=-1)

        # truncate back to J
        u2k = torch.zeros_like(u_hat)
        u2k[..., :self.J // 2] = U2_hat[..., :self.J // 2]
        u2k[..., self.J // 2 + 1:self.J] = U2_hat[..., self.J + 1:J_pad]
        return u2k  # = (u^2)_hat, length J, zeros at k=0 and k=J//2

    def dudt(self, u_hat: torch.Tensor) -> torch.Tensor:
        """RHS of tKdV in spectral space."""
        u2k = self._quadratic_dealiased(u_hat)
        nonlinear = -1j * self.k * self.h3_pde * u2k / 2.0  # (u^2/2)_hat * (-ik)
        dispersion = 1j * self.k ** 3 * self.h2_pde * u_hat
        return nonlinear + dispersion

    def _solve_midpoint_stage(
        self,
        a: torch.Tensor,
        w: float,
        z_init: torch.Tensor = None,
        n_iter: int = 1000,
        tol: float = 1e-12,
    ) -> torch.Tensor:
        """
        Solve one implicit midpoint stage y = a + w*dt*F(½(y+a)) via IMEX Picard.

        z = ½(y+a); z = M*(a + (tau/2)*N(z)), M = 1/(1 - (tau/2)*ik³ h2_pde).
        Returns y = 2z - a.
        """
        tau = w * self.dt
        Lk = 1j * self.k ** 3 * self.h2_pde
        M = 1.0 / (1.0 - (tau / 2.0) * Lk)

        z = a.clone() if z_init is None else z_init
        for _ in range(n_iter):
            N_z = -1j * self.k * self.h3_pde * self._quadratic_dealiased(z) / 2.0
            z_new = M * (a + (tau / 2.0) * N_z)
            err = (z_new - z).abs().pow(2).sum().item()
            z = z_new
            if err < tol:
                break

        return 2.0 * z - a

    def timestep(
        self,
        u_hat: torch.Tensor,
        nsteps: int = 1,
        ukm1: torch.Tensor = None,
        ukm2: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Advance u_hat by nsteps steps using the 4th-order symplectic scheme (McLachlan 1993).

        Three-stage scheme with w1=w3=(2+2^{1/3}+2^{-1/3})/3, w2=1-2w1.
        Warm-starts use MATLAB's history-based quadratic extrapolation (ukm1, ukm2).
        If ukm1/ukm2 not provided, defaults to u_hat (safe for first step).
        Post-step real round-trip kills imaginary roundoff and re-zeroes k=0 and Nyquist.
        """
        w1 = (2.0 + 2.0 ** (1.0 / 3.0) + 2.0 ** (-1.0 / 3.0)) / 3.0
        w2 = 1.0 - 2.0 * w1
        w3 = w1

        if ukm1 is None:
            ukm1 = u_hat.clone()
        if ukm2 is None:
            ukm2 = u_hat.clone()

        def interp(u0, um1, um2, w):
            # quadratic extrapolation matching MATLAB's interp lambda
            return (0.5 * w * (w + 3) + 1) * u0 - w * (w + 2) * um1 + 0.5 * w * (w + 1) * um2

        for _ in range(nsteps):
            z1_init = 0.5 * (u_hat + interp(u_hat, ukm1, ukm2, w1))
            y1 = self._solve_midpoint_stage(u_hat, w1, z_init=z1_init)
            z2_init = 0.5 * (y1 + interp(u_hat, ukm1, ukm2, w1 + w2))
            y2 = self._solve_midpoint_stage(y1, w2, z_init=z2_init)
            z3_init = 0.5 * (y2 + interp(u_hat, ukm1, ukm2, 1.0))
            ukm2 = ukm1
            ukm1 = u_hat
            u_hat = self._solve_midpoint_stage(y2, w3, z_init=z3_init)
            # project back to real and re-FFT: kills imaginary roundoff, zeroes k=0 and Nyquist
            u_hat = self.phys2spec(self.spec2phys(u_hat))

        return u_hat

    def hamiltonian(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute H = h3*H3 - h2*H2.

        H3 = (1/6) int u^3 dx  via Parseval over full spectrum (matches MATLAB solver).
        H2 = (1/2) int (u_x)^2 dx  via Parseval over full spectrum.
        """
        u2k = self._quadratic_dealiased(u_hat)  # = (u^2)_hat, length J
        # Parseval: int u^3 dx = 2*pi * real(sum_k (u^2)_hat[k] * conj(u_hat[k]))
        H3 = (1.0 / 6.0) * 2.0 * torch.pi * (u2k * u_hat.conj()).real.sum(dim=-1)

        # Parseval: int (u_x)^2 dx = 2*pi * sum_k k^2 |u_hat[k]|^2 (summing both +/- k)
        # = 2*pi * 0.5 * sum_{all k} k^2 |u_hat[k]|^2  ... but actually:
        # sum_{all k} k^2 |u_hat|^2 = 2 * sum_{k=1}^{J/2-1} k^2 |u_hat|^2  (by symmetry)
        # so H2 = pi * sum_{all k} k^2 |u_hat[k]|^2  = 2*pi * sum_{k=1}^{J/2-1} k^2 |u_hat[k]|^2
        H2 = torch.pi * (self.k ** 2 * u_hat.abs() ** 2).sum(dim=-1).real

        return self.h3 * H3 - self.h2 * H2

    def energy(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        L2 energy E(u) = 2*pi * 0.5 * sum_{k!=0} |u_hat[k]|^2.

        Sums both positive and negative k (indices 1..J-1); the 0.5 cancels the
        double-counting from conjugate symmetry of real u.
        Should remain = 1 throughout the simulation.
        """
        return 2.0 * torch.pi * 0.5 * (u_hat[..., 1:].abs() ** 2).sum(dim=-1).real

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

        Proposals are made in physical space to maintain Hermitian symmetry automatically.
        """
        device = self.k.device

        # initialize: random physical field → spectral → normalize to E=1
        u_phys = torch.randn(self.J, dtype=torch.float64, device=device)
        u_hat = self.phys2spec(u_phys)
        u_hat = u_hat / torch.sqrt(self.energy(u_hat))

        H_cur = self.hamiltonian(u_hat)

        samples = []
        n_total = n_burnin + n_samples * thin

        for step in range(n_total):
            # propose in physical space: adds noise, then back to spectral
            noise_phys = torch.randn(self.J, dtype=torch.float64, device=device)
            noise_hat = self.phys2spec(noise_phys)  # Hermitian symmetric, zeros at 0/Nyquist
            w_hat = u_hat + noise_hat / self.Lambda ** 0.5
            w_hat = w_hat / torch.sqrt(self.energy(w_hat))

            H_prop = self.hamiltonian(w_hat)

            log_alpha = -theta * (H_prop - H_cur)
            if torch.log(torch.rand(1, device=device)) < log_alpha:
                u_hat = w_hat
                H_cur = H_prop

            if step >= n_burnin and (step - n_burnin) % thin == 0:
                samples.append(u_hat.clone())

        samples = torch.stack(samples, dim=0)  # (n_samples, J)
        return samples.squeeze(0) if n_samples == 1 else samples

    def precomputed_initial_condition(
        self, folder: str, index: int, step: int = 0
    ) -> torch.Tensor:
        """Load a precomputed spectral state from {folder}/{index}_{step}.pt."""
        import os
        path = os.path.join(folder, f"{index}_{step}.pt")
        return torch.load(path, map_location=self.k.device)
