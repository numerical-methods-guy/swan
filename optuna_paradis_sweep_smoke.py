import os
import re
import time
import json
import queue
import argparse
import subprocess
from pathlib import Path

import optuna


def find_best_val_loss(exp_dir: Path) -> float:
    best = float("inf")
    if not exp_dir.exists():
        return best
    for ckpt in exp_dir.glob("version_*/checkpoints/*.ckpt"):
        m = re.search(r"val_loss=([0-9]*\.?[0-9]+)", ckpt.name)
        if m:
            best = min(best, float(m.group(1)))
    return best


def run_one_trial(
    python_bin: str,
    repo_dir: Path,
    config_path: Path,
    save_dir: Path,
    dt_solver: int,
    num_train_examples: int,
    seed: int,
    trial_number: int,
    gpu_id: int,
) -> float:
    # SMOKE TEST: fixed tiny run
    pretrain_epochs = 2
    finetune_epochs = 2

    exp_name = (
        f"SMOKE_paradis_dt{dt_solver}"
        f"_train{num_train_examples}"
        f"_pre{pretrain_epochs}_ft{finetune_epochs}"
        f"_seed{seed}"
        f"_t{trial_number}"
    )

    exp_dir = save_dir / exp_name
    trial_log_dir = save_dir / "optuna_logs_smoke"
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
        f.write(f"CMD: {' '.join(cmd)}\nGPU: {gpu_id}\nTIME: {time.ctime()}\n\n")
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
        return float("inf")

    best = find_best_val_loss(exp_dir)

    summary = {
        "exp_name": exp_name,
        "gpu": gpu_id,
        "dt_solver": dt_solver,
        "num_train_examples": num_train_examples,
        "pretrain_epochs": pretrain_epochs,
        "finetune_epochs": finetune_epochs,
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

    ap.add_argument("--n_trials", type=int, default=4)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--study_name", type=str, default="SMOKE_paradis_dt75")
    ap.add_argument("--storage", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    config_path = (repo_dir / args.config).resolve()
    save_dir = (repo_dir / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    gpu_queue = queue.Queue()
    for gid in range(args.n_jobs):
        gpu_queue.put(gid)

    def objective(trial: optuna.Trial) -> float:
        # SMOKE: just vary num_train_examples to ensure overrides + logging work.
        num_train_examples = trial.suggest_categorical(
            "data.num_train_examples", [256, 512, 1024, 2048]
        )
        seed = args.seed

        gpu_id = gpu_queue.get()
        try:
            val = run_one_trial(
                python_bin=args.python_bin,
                repo_dir=repo_dir,
                config_path=config_path,
                save_dir=save_dir,
                dt_solver=args.dt_solver,
                num_train_examples=num_train_examples,
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

    out = {"best_value": study.best_value, "best_params": study.best_params}
    out_path = save_dir / "optuna_best_smoke.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("==== OPTUNA SMOKE DONE ====")
    print(out)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

