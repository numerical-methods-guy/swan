import os
import re
import time
import json
import queue
import argparse
import subprocess
import threading
import heapq
from pathlib import Path

import optuna


TODAY_YYYYMMDD = time.strftime("%Y%m%d")


def find_best_val_loss(exp_dir: Path) -> float:
    best = float("inf")
    if not exp_dir.exists():
        return best

    for ckpt in exp_dir.glob("version_*/checkpoints/*.ckpt"):
        name = ckpt.name

        m = re.search(r"val_loss=([0-9]*\.?[0-9]+)", name)
        if m:
            val = float(m.group(1))
            best = min(best, val)
            continue

        m = re.search(r"-([0-9]*\.?[0-9]+)\.ckpt$", name)
        if m:
            val = float(m.group(1))
            best = min(best, val)
            continue

    return best


def run_one_trial(
    python_bin: str,
    repo_dir: Path,
    train_script: Path,
    config_path: Path,
    save_dir: Path,
    dt_solver: int,
    num_train_examples: int,
    seed: int,
    trial_number: int,
    gpu_id: int,
    hidden_dim: int,
    num_layers: int,
    num_encoder_layers: int,
    num_vels: int,
    diffusion_size: int,
    reaction_size: int,
    bias_channels: int,
    learning_rate: float,
) -> tuple[float, str]:
    pretrain_epochs = 75
    finetune_epochs = 0

    exp_name = (
        f"{TODAY_YYYYMMDD}_paradis_dt{dt_solver}"
        f"_train{num_train_examples}"
        f"_pt{pretrain_epochs}"
        f"_ft{finetune_epochs}"
        f"_hd{hidden_dim}"
        f"_L{num_layers}"
        f"_enc{num_encoder_layers}"
        f"_vel{num_vels}"
        f"_diff{diffusion_size}"
        f"_react{reaction_size}"
        f"_bias{bias_channels}"
        f"_lr{learning_rate:.2e}"
        f"_seed{seed}"
        f"_t{trial_number:05d}"
    )

    exp_dir = save_dir / exp_name

    trial_log_dir = save_dir / "optuna_logs"
    trial_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = trial_log_dir / f"{exp_name}.log"

    cmd = [
        python_bin,
        str(train_script),
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
        "--model.paradis.hidden_dim",
        str(hidden_dim),
        "--model.paradis.num_layers",
        str(num_layers),
        "--model.paradis.num_encoder_layers",
        str(num_encoder_layers),
        "--model.paradis.num_vels",
        str(num_vels),
        "--model.paradis.diffusion_size",
        str(diffusion_size),
        "--model.paradis.reaction_size",
        str(reaction_size),
        "--model.paradis.bias_channels",
        str(bias_channels),
        "--training.learning_rate",
        str(learning_rate),
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
        summary = {
            "exp_name": exp_name,
            "gpu": gpu_id,
            "dt_solver": dt_solver,
            "num_train_examples": num_train_examples,
            "pretrain_epochs": pretrain_epochs,
            "finetune_epochs": finetune_epochs,
            "seed": seed,
            "best_val_loss": float("inf"),
            "status": "FAILED",
            "log_file": str(log_file),
            "exp_dir": str(exp_dir),
        }
        with open(trial_log_dir / f"{exp_name}.json", "w") as jf:
            json.dump(summary, jf, indent=2)
        return float("inf"), exp_name

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
        "status": "OK",
        "log_file": str(log_file),
        "exp_dir": str(exp_dir),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_encoder_layers": num_encoder_layers,
        "num_vels": num_vels,
        "diffusion_size": diffusion_size,
        "reaction_size": reaction_size,
        "bias_channels": bias_channels,
        "learning_rate": learning_rate,
    }
    with open(trial_log_dir / f"{exp_name}.json", "w") as jf:
        json.dump(summary, jf, indent=2)

    return best, exp_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_dir", type=str, required=True)
    ap.add_argument("--python_bin", type=str, required=True)
    ap.add_argument("--config", type=str, default="config_paradis.yaml")
    ap.add_argument("--save_dir", type=str, default="trial_optuna")
    ap.add_argument("--dt_solver", type=int, default=30)
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--study_name", type=str, default="paradis_arch_lr_sweep_dt30_pt75")
    ap.add_argument("--storage", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    python_bin = args.python_bin
    config_path = (repo_dir / args.config).resolve()
    save_dir = (repo_dir / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    train_script = repo_dir / "avi_train_gbell_train_gbell_val_2.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    num_train_examples = 1024

    gpu_queue = queue.Queue()
    for gid in range(args.n_jobs):
        gpu_queue.put(gid)

    CAND_HIDDEN_DIM = [24, 32, 48, 64]
    CAND_NUM_LAYERS = [5, 6, 7, 8]
    CAND_NUM_ENCODER_LAYERS = [2, 3, 4, 5]
    CAND_NUM_VELS = [8, 10, 12, 14]
    CAND_DIFFUSION_SIZE = [12, 16, 20, 24]
    CAND_REACTION_SIZE = [8, 12, 16, 20]
    CAND_BIAS_CHANNELS = [3, 4, 6, 8]
    CAND_LR = [1.2e-3, 1.5e-3, 2.0e-3, 3.0e-3]

    ranking_lock = threading.Lock()
    ranking_heap: list[tuple[float, str]] = []
    ranking_path = save_dir / "optuna_ranking.json"

    def _update_ranking(val: float, exp_name: str) -> None:
        with ranking_lock:
            heapq.heappush(ranking_heap, (float(val), str(exp_name)))
            ordered = sorted(ranking_heap, key=lambda x: x[0])
            payload = [{"best_val_loss": v, "exp_name": n} for v, n in ordered]
            with open(ranking_path, "w") as f:
                json.dump(payload, f, indent=2)

    def objective(trial: optuna.Trial) -> float:
        hidden_dim = trial.suggest_categorical("model.paradis.hidden_dim", CAND_HIDDEN_DIM)
        num_layers = trial.suggest_categorical("model.paradis.num_layers", CAND_NUM_LAYERS)
        num_encoder_layers = trial.suggest_categorical(
            "model.paradis.num_encoder_layers", CAND_NUM_ENCODER_LAYERS
        )
        num_vels = trial.suggest_categorical("model.paradis.num_vels", CAND_NUM_VELS)
        diffusion_size = trial.suggest_categorical("model.paradis.diffusion_size", CAND_DIFFUSION_SIZE)
        reaction_size = trial.suggest_categorical("model.paradis.reaction_size", CAND_REACTION_SIZE)
        bias_channels = trial.suggest_categorical("model.paradis.bias_channels", CAND_BIAS_CHANNELS)
        learning_rate = trial.suggest_categorical("training.learning_rate", CAND_LR)

        seed = args.seed

        gpu_id = gpu_queue.get()
        try:
            val, exp_name = run_one_trial(
                python_bin=python_bin,
                repo_dir=repo_dir,
                train_script=train_script,
                config_path=config_path,
                save_dir=save_dir,
                dt_solver=args.dt_solver,
                num_train_examples=num_train_examples,
                seed=seed,
                trial_number=trial.number,
                gpu_id=gpu_id,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_encoder_layers=num_encoder_layers,
                num_vels=num_vels,
                diffusion_size=diffusion_size,
                reaction_size=reaction_size,
                bias_channels=bias_channels,
                learning_rate=learning_rate,
            )
            trial.set_user_attr("gpu_id", gpu_id)
            trial.set_user_attr("exp_name", exp_name)
            trial.set_user_attr("best_val_loss", float(val))
            _update_ranking(val, exp_name)
            return float(val)
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

    ordered = []
    if ranking_path.exists():
        with open(ranking_path, "r") as f:
            ordered = json.load(f)

    out = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "study_name": args.study_name,
        "dt_solver": args.dt_solver,
        "num_train_examples": num_train_examples,
        "pretrain_epochs": 75,
        "finetune_epochs": 0,
        "ranking_file": str(ranking_path),
        "top_results": ordered[: min(50, len(ordered))],
    }
    out_path = save_dir / "optuna_best.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

