import os
import json
import argparse
import datetime
import subprocess

import torch

from kdv.tkdv_solver import TKdVSolver


def build_solver(J, dt_solver, D, E0, L0, device):
    return TKdVSolver(J=J, dt=dt_solver, D=D, E0=E0, L0=L0).to(device).double()


def make_output_folder(ictype, dt, dt_solver, base_dir=None):
    date_str = datetime.date.today().strftime("%Y%m%d")
    folder_name = f"tkdv_{ictype}_dt{dt}_dtsolver{dt_solver}_{date_str}"
    folder_path = os.path.join(base_dir or "kdv/Saved_Datasets", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, "stability_check"), exist_ok=True)
    return folder_path


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unavailable"


def save_trajectories(
    solver,
    dt,
    D,
    E0,
    L0,
    ictype,
    n_samples,
    n_steps_per_trajectory,
    output_folder,
    theta,
    gibbs_burnin,
    gibbs_thin,
    dt_solver_ref,
    n_stability_samples,
    n_stability_steps,
    stability_threshold,
):
    """Generate tKdV trajectories, save spectral states as .pt files,
    compute normalization statistics online, and run a stability check
    by comparing against a reference solver with smaller dt."""

    stability_dir = os.path.join(output_folder, "stability_check")

    # build reference solver with finer timestep
    ref_solver = TKdVSolver(
        J=solver.J,
        dt=dt_solver_ref,
        D=D,
        E0=E0,
        L0=L0,
    ).to(solver.k.device).double()
    # copy physical param buffers from main solver so they match exactly
    ref_solver.h3_pde = solver.h3_pde
    ref_solver.h2_pde = solver.h2_pde
    ref_solver.h3 = solver.h3
    ref_solver.h2 = solver.h2

    # nsteps: solver sub-steps per model timestep; n_ref_substeps: ref sub-steps per model timestep
    nsteps = max(1, round(dt / solver.dt))
    n_ref_substeps = max(1, round(dt / dt_solver_ref))

    # Chan's parallel Welford accumulators — scalar mean/var over all grid points and steps
    field_count = 0
    field_mean  = None
    field_M2    = None
    field_min   = None
    field_max   = None

    stability_errors = []

    with torch.no_grad():
        for i in range(n_samples):
            if ictype == "gibbs":
                u_hat = solver.gibbs_initial_condition(
                    theta=theta,
                    n_burnin=gibbs_burnin,
                    n_samples=1,
                    thin=gibbs_thin,
                )
            else:
                raise ValueError(f"Unsupported ictype: {ictype!r}")

            do_stability = i < n_stability_samples
            if do_stability:
                u_hat_ref = u_hat.clone()

            for step in range(n_steps_per_trajectory + 1):
                path = os.path.join(output_folder, f"{i}_{step}.pt")
                torch.save(u_hat.cpu(), path)

                u = solver.spec2phys(u_hat).double()
                batch_n    = u.shape[-1]
                batch_mean = u.mean()
                batch_M2   = u.var(unbiased=False) * batch_n

                if field_count == 0:
                    field_mean  = batch_mean
                    field_M2    = batch_M2
                    field_count = batch_n
                else:
                    new_count   = field_count + batch_n
                    delta       = batch_mean - field_mean
                    field_mean  = field_mean + delta * batch_n / new_count
                    field_M2    = field_M2 + batch_M2 + delta ** 2 * field_count * batch_n / new_count
                    field_count = new_count

                if field_min is None:
                    field_min = u.min()
                    field_max = u.max()
                else:
                    field_min = torch.minimum(field_min, u.min())
                    field_max = torch.maximum(field_max, u.max())

                # stability check: compare main vs ref at each step
                if do_stability and step > 0 and step <= n_stability_steps:
                    ref_path = os.path.join(stability_dir, f"{i}_{step}_ref.pt")
                    torch.save(u_hat_ref.cpu(), ref_path)

                    u_main = solver.spec2phys(u_hat).double()
                    u_ref  = ref_solver.spec2phys(u_hat_ref).double()
                    diff    = (u_main - u_ref).norm()
                    denom   = u_ref.norm().clamp(min=1e-8)
                    rel_err = (diff / denom).item()
                    stability_errors.append({"sample": i, "step": step, "rel_l2_error": rel_err})
                    print(f"  [stability] sample {i}, step {step}: rel L2 error = {rel_err:.6f}")

                if step < n_steps_per_trajectory:
                    u_hat = solver.timestep(u_hat, nsteps=nsteps)
                    if do_stability and step < n_stability_steps:
                        u_hat_ref = ref_solver.timestep(u_hat_ref, nsteps=n_ref_substeps)

            if (i + 1) % 10 == 0:
                print(f"  saved trajectory {i + 1}/{n_samples}")

    field_var = (field_M2 / field_count).clamp(min=1e-20)

    stats = {
        "field_mean": field_mean.item(),
        "field_var":  field_var.item(),
        "field_std":  field_var.item() ** 0.5,
        "field_min":  field_min.item(),
        "field_max":  field_max.item(),
    }

    if stability_errors:
        max_err  = max(e["rel_l2_error"] for e in stability_errors)
        mean_err = sum(e["rel_l2_error"] for e in stability_errors) / len(stability_errors)
        passed   = max_err < stability_threshold
        print(f"\nStability check {'PASSED' if passed else 'FAILED'}  "
              f"(max rel L2 error = {max_err:.6f}, threshold = {stability_threshold})")
    else:
        max_err = mean_err = 0.0
        passed = True

    stability_summary = {
        "n_stability_samples":  n_stability_samples,
        "n_stability_steps":    n_stability_steps,
        "dt_solver_ref":        dt_solver_ref,
        "n_ref_substeps":       n_ref_substeps,
        "stability_threshold":  stability_threshold,
        "max_rel_l2_error":     max_err,
        "mean_rel_l2_error":    mean_err,
        "passed":               passed,
        "per_sample_errors":    stability_errors,
    }

    return stats, stability_summary


def save_metadata(output_folder, args, stats, stability_summary):
    nsteps = max(1, round(args.dt / args.dt_solver))
    meta = {
        "ictype":                 args.ictype,
        "dt":                     args.dt,
        "dt_solver":              args.dt_solver,
        "nsteps":                 nsteps,
        "dt_solver_ref":          args.dt_solver_ref,
        "J":                      args.J,
        "D":                      args.D,
        "E0":                     args.E0,
        "L0":                     args.L0,
        "theta":                  args.theta,
        "gibbs_burnin":           args.gibbs_burnin,
        "gibbs_thin":             args.gibbs_thin,
        "n_samples":              args.n_samples,
        "n_steps_per_trajectory": args.n_steps_per_trajectory,
        "date":                   datetime.date.today().isoformat(),
        "timestamp":              datetime.datetime.now().isoformat(),
        "git_hash":               get_git_hash(),
        "field_mean":             stats["field_mean"],
        "field_var":              stats["field_var"],
        "field_std":              stats["field_std"],
        "field_min":              stats["field_min"],
        "field_max":              stats["field_max"],
        "stability_check":        stability_summary,
    }

    json_path = os.path.join(output_folder, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    sc = stability_summary
    txt_lines = [
        "=" * 60,
        "tKdV Precomputed Dataset Metadata",
        "=" * 60,
        f"  IC type                  : {meta['ictype']}",
        f"  Model timestep (dt)      : {meta['dt']}",
        f"  Solver sub-step          : {meta['dt_solver']}",
        f"  Sub-steps per model step : {meta['nsteps']}",
        f"  Reference sub-step       : {meta['dt_solver_ref']}",
        f"  Grid points (J)          : {meta['J']}",
        f"  D                        : {meta['D']}",
        f"  E0                       : {meta['E0']}",
        f"  L0                       : {meta['L0']}",
        f"  theta                    : {meta['theta']}",
        f"  Gibbs burn-in            : {meta['gibbs_burnin']}",
        f"  Gibbs thin               : {meta['gibbs_thin']}",
        f"  Number of samples        : {meta['n_samples']}",
        f"  Steps per trajectory     : {meta['n_steps_per_trajectory']}  (step 0 = IC)",
        f"  Date                     : {meta['date']}",
        f"  Timestamp                : {meta['timestamp']}",
        f"  Git hash                 : {meta['git_hash']}",
        "",
        "Normalization stats:",
        f"  field_mean : {meta['field_mean']:.6f}",
        f"  field_var  : {meta['field_var']:.6f}",
        f"  field_std  : {meta['field_std']:.6f}",
        f"  field_min  : {meta['field_min']:.6f}",
        f"  field_max  : {meta['field_max']:.6f}",
        "",
        "Stability check:",
        f"  dt_solver_ref            : {sc['dt_solver_ref']}",
        f"  Ref sub-steps per step   : {sc['n_ref_substeps']}",
        f"  Samples checked          : {sc['n_stability_samples']}",
        f"  Steps checked            : {sc['n_stability_steps']}",
        f"  Threshold                : {sc['stability_threshold']}",
        f"  Max rel L2 error         : {sc['max_rel_l2_error']:.6f}",
        f"  Mean rel L2 error        : {sc['mean_rel_l2_error']:.6f}",
        f"  Result                   : {'PASSED' if sc['passed'] else 'FAILED'}",
        "=" * 60,
    ]
    txt_path = os.path.join(output_folder, "metadata.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines) + "\n")

    print(f"\nMetadata saved to {json_path} and {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and save precomputed tKdV trajectory datasets."
    )
    parser.add_argument("--ictype", type=str, default="gibbs", choices=["gibbs"],
                        help="Initial condition type")
    parser.add_argument("--J", type=int, default=32,
                        help="Number of grid points")
    parser.add_argument("--dt", type=float, default=1.0,
                        help="Model timestep (one step saved per trajectory)")
    parser.add_argument("--dt_solver", type=float, default=0.01,
                        help="Integrator sub-step passed to the solver (dt_solver < dt)")
    parser.add_argument("--dt_solver_ref", type=float, default=0.001,
                        help="Reference solver sub-step for stability check (should be smaller than dt_solver)")
    parser.add_argument("--D", type=float, default=1.0)
    parser.add_argument("--E0", type=float, default=100.0)
    parser.add_argument("--L0", type=float, default=6.0)
    parser.add_argument("--theta", type=float, default=-0.1,
                        help="Inverse temperature for Gibbs measure")
    parser.add_argument("--gibbs_burnin", type=int, default=1000,
                        help="Number of MCMC burn-in steps")
    parser.add_argument("--gibbs_thin", type=int, default=10,
                        help="MCMC thinning interval")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of trajectories to generate")
    parser.add_argument("--n_steps_per_trajectory", type=int, default=5,
                        help="Number of timesteps per trajectory (step 0 is the IC)")
    parser.add_argument("--n_stability_samples", type=int, default=10,
                        help="Number of samples to use for the stability check")
    parser.add_argument("--n_stability_steps", type=int, default=2,
                        help="Number of rollout steps to compare in the stability check")
    parser.add_argument("--stability_threshold", type=float, default=0.01,
                        help="Relative L2 error threshold for stability pass/fail")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Base directory for output (default: kdv/Saved_Datasets)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"\nBuilding solver (J={args.J}, dt_solver={args.dt_solver}, D={args.D}, E0={args.E0}, L0={args.L0})...")
    solver = build_solver(args.J, args.dt_solver, args.D, args.E0, args.L0, device)

    output_folder = make_output_folder(args.ictype, args.dt, args.dt_solver, base_dir=args.save_dir)
    print(f"Output folder: {output_folder}")

    nsteps = max(1, round(args.dt / args.dt_solver))
    print(f"\nGenerating {args.n_samples} trajectories ({args.n_steps_per_trajectory} steps each, "
          f"dt={args.dt}, dt_solver={args.dt_solver}, nsteps={nsteps})...")
    stats, stability_summary = save_trajectories(
        solver=solver,
        dt=args.dt,
        D=args.D,
        E0=args.E0,
        L0=args.L0,
        ictype=args.ictype,
        n_samples=args.n_samples,
        n_steps_per_trajectory=args.n_steps_per_trajectory,
        output_folder=output_folder,
        theta=args.theta,
        gibbs_burnin=args.gibbs_burnin,
        gibbs_thin=args.gibbs_thin,
        dt_solver_ref=args.dt_solver_ref,
        n_stability_samples=args.n_stability_samples,
        n_stability_steps=args.n_stability_steps,
        stability_threshold=args.stability_threshold,
    )

    save_metadata(output_folder, args, stats, stability_summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
