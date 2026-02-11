import os
import re
import sys
import time
import json
import queue
import shutil
import argparse
import subprocess
from pathlib import Path

import optuna


def find_best_val_loss(exp_dir: Path) -> float:
    """
    Find smallest val_loss from checkpoint filenames under:
      exp_dir/version_*/checkpoints/*.ckpt
    Returns +inf if nothing found.
    """
    best = float("inf")
    if not exp_dir.exists():
        return best

    for ckpt in exp_dir.glob("version_*/checkpoints/*.ckpt"):
        m = re.search(r"val_loss=([0-9]*\.?[0-9]+)", ckpt.name)
        if m:
            val = float(m.group(1))
            if val < best:
                best = val
    return best


def run_one_trial(
    python_bin: str,
    repo_dir: Path,
    config_path: Path,
    save_dir: Path,
    dt_solver: int,
    num_train_examples: int,
    finetune_epochs: int,
    seed: int,
    trial_number: int,
    gpu_id: int,
) -> float:
    """
    Runs train.py once with overrides, pinned to one GPU via CUDA_VISIBLE_DEVICES.
    Returns best val_loss parsed from checkpoint names.
    """
    pretrain_epochs = 94 - finetune_epochs
    if pretrain_epochs < 0:
        raise ValueError("finetune_epochs cannot exceed 100.")

    exp_name = (
        f"paradis_dt{dt_solver}_optuna"
        f"_train{num_train_examples}"
        f"_ft{finetune_epochs}"
        f"_seed{seed}"
        f"_t{trial_number}"
    )

    exp_dir = save_dir / exp_name

    # Per-trial stdout/stderr log
    trial_log_dir = save_dir / "optuna_logs"
    trial_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = trial_log_dir / f"{exp_name}.log"

    cmd = [
        python_bin,
        str(repo_dir / "train.py"),
        "--config",
        str(config_path),
        "--experiment.name",
        exp_name,
        "--experiment.seed",
        str(seed),
        "--training.save_dir",
        str(save_dir),
        "--training.pretrain_epochs",
        str(pretrain_epochs),
        "--training.finetune_epochs",
        str(finetune_epochs),
        "--data.dt_solver",
        str(dt_solver),
        "--data.num_train_examples",
        str(num_train_examples),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with open(log_file, "w") as f:
        f.write(f"CMD: {' '.join(cmd)}\n")
        f.write(f"GPU: {gpu_id}\n")
        f.write(f"TIME: {time.ctime()}\n\n")
        f.flush()

        p = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if p.returncode != 0:
        # If training fails, make it obvious to Optuna (bad objective value)
        return float("inf")

    best = find_best_val_loss(exp_dir)

    # Save a tiny json summary for convenience
    summary = {
        "exp_name": exp_name,
        "gpu": gpu_id,
        "dt_solver": dt_solver,
        "num_train_examples": num_train_examples,
        "finetune_epochs": finetune_epochs,
        "pretrain_epochs": pretrain_epochs,
        "seed": seed,
        "best_val_loss": best,
        "log_file": str(log_file),
        "exp_dir": str(exp_dir),
    }
    with open(trial_log_dir / f"{exp_name}.json", "w") as jf:
        json.dump(summary, jf, indent=2)

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_dir", type=str, required=True)
    ap.add_argument("--python_bin", type=str, required=True)
    ap.add_argument("--config", type=str, default="config_paradis.yaml")
    ap.add_argument("--save_dir", type=str, default="trial")
    ap.add_argument("--dt_solver", type=int, default=75)

    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--n_jobs", type=int, default=4)  # how many GPUs to use in parallel
    ap.add_argument("--study_name", type=str, default="paradis_dt75_train_ft_sweep")
    ap.add_argument("--storage", type=str, default=None)  # e.g. sqlite:///trial/optuna.db
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    python_bin = args.python_bin
    config_path = (repo_dir / args.config).resolve()
    save_dir = (repo_dir / args.save_dir).resolve()

    save_dir.mkdir(parents=True, exist_ok=True)

    # GPU pool: assumes the PBS job requested 4 GPUs and they appear as 0,1,2,3
    gpu_queue = queue.Queue()
    for gid in range(args.n_jobs):
        gpu_queue.put(gid)

    def objective(trial: optuna.Trial) -> float:
        # Search space (exactly what you asked)
        num_train_examples = trial.suggest_categorical(
            "data.num_train_examples", [256, 512, 768]
        )
        finetune_epochs = trial.suggest_categorical(
            "training.finetune_epochs", [0, 10, 20]
        )

        # Optional: allow Optuna to vary seed a bit, but default to 42 unless you want otherwise
        seed = args.seed

        gpu_id = gpu_queue.get()  # block until a GPU is free
        try:
            val = run_one_trial(
                python_bin=python_bin,
                repo_dir=repo_dir,
                config_path=config_path,
                save_dir=save_dir,
                dt_solver=args.dt_solver,
                num_train_examples=num_train_examples,
                finetune_epochs=finetune_epochs,
                seed=seed,
                trial_number=trial.number,
                gpu_id=gpu_id,
            )
            trial.set_user_attr("gpu_id", gpu_id)
            return val
        finally:
            gpu_queue.put(gpu_id)

    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    # Write final best result
    out = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "study_name": args.study_name,
    }
    out_path = save_dir / "optuna_best.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("==== OPTUNA DONE ====")
    print(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

