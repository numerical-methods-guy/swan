import os, uuid, pathlib
from pathlib import Path
import math
import shutil
import tempfile
import logging
from multiprocessing import Process, set_start_method

import hydra
import argparse
import optuna
from omegaconf import DictConfig
from optuna.samplers import TPESampler

from trainer import LitParadis
from data.datamodule import Era5DataModule
from utils.system import setup_system
from utils.callbacks import enable_callbacks

from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import lightning as L

#  Optuna shared storage (SQLite with longer timeout helps concurrency)
STORAGE_URL = "sqlite:////data/logs_optuna/paradis_5deg.db?timeout=60"
GLOBAL_MAX_TRIALS = 1000

# Environment variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

MAX_LOSS_SPIKE = 1.2


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_loss_spike(arr, *, warmup=0, eps=1e-12) -> float:
    import numpy as np

    a = np.asarray(arr, dtype=float)
    if warmup:
        a = a[warmup:]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    rmin = np.minimum.accumulate(a)
    return float(np.nanmax(a / (rmin + eps)))  # ≥1.0


def constraint_check(trial):
    import numpy as np

    spike = trial.user_attrs.get("loss_spike")
    if spike is None:
        # Presumably an incomplete trial, marked as failure before
        # setting these attributes.  Optimistically assume it satisfies
        # the constraints
        return (-1.0,)

    # When no spikes exist, make sure it's considered feasible
    if spike <= 1.0 + 1e-9:
        return (-1.0,)

    # Otherwise compute the spike metric
    return (np.log((max(spike, 1.0) - 1) / (MAX_LOSS_SPIKE - 1)),)


class CurveRecorder(L.Callback):
    def __init__(self, key="val_loss"):
        self.key = key
        self.values = []
        self.steps = []

    def on_validation_end(self, trainer, pl_module):
        x = trainer.callback_metrics.get(self.key)
        if x is None:
            return
        v = float(x.detach().item() if hasattr(x, "detach") else x)
        step = int(trainer.global_step)
        self.values.append(v)
        self.steps.append(step)


def make_objective(
    study_name: str,
    cfg: DictConfig,
    worker_tag: str,
    log_root: str = "/data/logs_optuna/",
):
    def objective(trial: optuna.trial.Trial) -> float:

        # Per-trial sandbox directory
        trial_tmp = tempfile.mkdtemp(prefix=f"trial_{trial.number}_")
        try:
            # Scope compile/runtime caches to THIS trial (keeps things tidy)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(trial_tmp) / "inductor")
            os.environ["TRITON_CACHE_DIR"] = str(Path(trial_tmp) / "triton")
            os.environ["CUDA_CACHE_PATH"] = str(Path(trial_tmp) / "cuda")
            os.environ["XDG_CACHE_HOME"] = str(Path(trial_tmp) / "xdg")
            for d in ("inductor", "triton", "cuda", "xdg"):
                Path(trial_tmp, d).mkdir(exist_ok=True)

            # Delay these imports until the process is pinned to a single GPU
            import torch
            import lightning as L
            from lightning.pytorch.loggers import TensorBoardLogger

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.compiler.reset()

            # Reset the compilation cache to avoid hitting recompile limits
            torch._dynamo.config.recompile_limit = 100000
            torch._dynamo.config.cache_size_limit = 100000

            # Search space
            latent_size = trial.suggest_int("latent_size", 64, 2048)
            reaction_size = trial.suggest_int("reaction_size", 64, 2048)
            velocity_vectors = trial.suggest_int("velocity_vectors", 64, 2048)
            num_layers = trial.suggest_int("num_layers", 2, 20)
            learning_rate = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
            bias_channels = trial.suggest_int("bias_channels", 1, 20)

            # Apply to configuration
            cfg.model.adv_interpolation = "bicubic"
            cfg.model.num_layers = num_layers
            cfg.model.latent_size = latent_size
            cfg.model.diffusion_size = latent_size
            cfg.model.reaction_size = reaction_size
            cfg.model.bias_channels = bias_channels
            cfg.model.velocity_vectors = velocity_vectors
            cfg.training.optimizer.lr = learning_rate
            cfg.dataset.n_time_inputs = 2

            # Setup model
            setup_system(cfg)
            datamodule = Era5DataModule(cfg)
            datamodule.setup(stage="fit")
            litmodel = LitParadis(datamodule, cfg)

            # Get a unique name for the running version
            run_name = f"trial_{trial.number:05d}-{worker_tag}"
            tb = TensorBoardLogger(save_dir=log_root, version=run_name)

            # Determine the model number of parameters
            total_params, trainable_params = count_parameters(litmodel)

            # Log them on both optuna and tensorboard
            trial.set_user_attr("params_total", total_params)
            trial.set_user_attr("params_trainable", trainable_params)

            tb.experiment.add_scalar("model/params_total", total_params, 0)
            tb.experiment.add_scalar("model/params_trainable", trainable_params, 0)
            tb.log_hyperparams(
                {"params_total": total_params, "params_trainable": trainable_params}
            )

            # Prepare callbacks
            callbacks = enable_callbacks(cfg)

            # Keep track of losses
            curve_rec_callback = CurveRecorder()
            callbacks.append(curve_rec_callback)

            # Monitoring and pruning
            monitor_metric = "val_loss"

            trainer = L.Trainer(
                default_root_dir=f"/data/logs_optuna/",
                accelerator=cfg.compute.accelerator,
                devices=1,  # exactly 1 GPU for this process
                num_nodes=1,
                strategy="auto",
                max_epochs=cfg.training.max_epochs,
                max_steps=cfg.training.max_steps,
                gradient_clip_val=cfg.training.gradient_clip_val,
                gradient_clip_algorithm="norm",
                log_every_n_steps=cfg.training.log_every_n_steps,
                callbacks=callbacks,
                precision="bf16-mixed" if cfg.compute.use_amp else "32-true",
                enable_progress_bar=cfg.training.progress_bar
                and not cfg.training.print_losses,
                enable_model_summary=True,
                logger=tb,
                val_check_interval=cfg.training.validation_dataset.validation_every_n_steps,
                limit_val_batches=cfg.training.validation_dataset.validation_batches,
                enable_checkpointing=cfg.training.checkpointing.enabled,
                accumulate_grad_batches=4,
                num_sanity_val_steps=0,  # speed up HPO
            )

            try:
                trainer.fit(litmodel, datamodule=datamodule)
                trial.set_user_attr("oom", False)
                loss_spike = get_loss_spike(curve_rec_callback.values)
                trial.set_user_attr("loss_spike", loss_spike)

                # Log maximum CUDA memory usage
                trial.set_user_attr(
                    "max_mem_mb", torch.cuda.max_memory_allocated() // 2**20
                )
            except torch.cuda.OutOfMemoryError:
                trial.set_user_attr("oom", True)
                trial.set_user_attr("loss_spike", 100)
                trial.set_user_attr("max_mem_mb", 1e16)
                raise optuna.TrialPruned("OOM during fit")

            finally:
                # free mem between trials
                del litmodel, datamodule
                if "torch" in globals() and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Log min_dt
            min_dt = trainer.callback_metrics.get("min_dt")
            min_dt = float(
                min_dt.detach().item() if hasattr(min_dt, "detach") else min_dt
            )
            trial.set_user_attr("min_dt", min_dt)

            # Log GZ500
            gz500name = "geopotential_h500"
            gz500 = trainer.callback_metrics.get(gz500name, None)
            gz500 = float(gz500.detach().item() if hasattr(gz500, "detach") else gz500)
            trial.set_user_attr("geopotential_h500", gz500)

            # Get monitor metric
            val_loss = trainer.callback_metrics.get(monitor_metric)
            if val_loss is None:
                # If not logged, prune to avoid poisoning the study
                raise optuna.TrialPruned(f"{monitor_metric} not logged")
            v = float(
                val_loss.detach().item() if hasattr(val_loss, "detach") else val_loss
            )

            trial.set_user_attr("validation_loss", v)

            if v > 1:
                raise optuna.TrialPruned(
                    "Validation value exceeds divergence threshold"
                )

            return v, trainable_params

        finally:
            shutil.rmtree(trial_tmp, ignore_errors=True)

    return objective


def _worker(
    cfg: DictConfig, gpu_ind: int, trials_for_this_worker: int, study_name: str
):
    """
    One worker process: makes only one GPU visible, then runs sequential trials.
    """
    # Pin process to a single GPU BEFORE any torch import in this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ind)

    # Optional: set device(0) inside the process (safe since only 1 visible GPU)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    except Exception:
        pass

    pruner = optuna.pruners.MedianPruner(
        # n_startup_trials=5,
        # n_warmup_steps=0, interval_steps=1
    )

    sampler = TPESampler(
        multivariate=True,
        n_startup_trials=10,
        constant_liar=True,
        constraints_func=constraint_check,
    )

    study = optuna.create_study(
        study_name=study_name,
        directions=["minimize", "minimize"],
        pruner=pruner,
        sampler=sampler,
        storage=STORAGE_URL,
        load_if_exists=True,
    )

    objective = make_objective(
        study_name,
        cfg,
        worker_tag=f"gpu{gpu_ind}",
        log_root=f"/data/logs_optuna/{study_name}",
    )

    study.optimize(
        objective,
        n_trials=trials_for_this_worker,
        n_jobs=1,
        gc_after_trial=True,
        callbacks=[MaxTrialsCallback(GLOBAL_MAX_TRIALS, states=(TrialState.COMPLETE,))],
    )


def _parse_visible_gpu_ids() -> list[int]:
    """
    Determine which GPU indices to use for workers.
    - If CUDA_VISIBLE_DEVICES is set (e.g., "0,1,2,3"), use its indices.
    - Else fallback to range(int(os.getenv("N_WORKERS", 1))).
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        try:
            return [int(x.strip()) for x in cvd.split(",") if x.strip() != ""]
        except Exception:
            pass
    # Fallback: allow override via env var
    n = int(os.environ.get("N_WORKERS", "1"))
    return list(range(n))


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, type=str)
    ap.add_argument("--prediction_res", required=True, type=int)
    args = ap.parse_args()

    study_name = args.study

    with hydra.initialize(config_path="config/", version_base=None):
        cfg = hydra.compose(config_name="paradis_settings")

    cfg.dataset.root_dir = "/scratch/era5_5.625deg_13level_compressed/"
    cfg.dataset.prediction_delta = str(args.prediction_res) + "h"

    cfg.model.base_dt = args.prediction_res * 3600
    cfg.model.forecast_steps = 1
    cfg.model.splitting = "lie-midpoint"
    cfg.model.residual = False
    cfg.model.incremental = True

    cfg.compute.num_devices = 1
    cfg.compute.num_nodes = 1
    cfg.compute.compile = True
    cfg.compute.use_amp = True
    cfg.compute.batch_size = 32
    cfg.compute.num_workers = 4

    cfg.normalization.standard = True

    cfg.training.max_epochs = 30
    cfg.training.log_every_n_steps = 50
    cfg.training.checkpointing.enabled = False
    cfg.training.early_stopping.enabled = True

    cfg.training.dataset.start_date = "2010-01-01"
    cfg.training.dataset.end_date = "2014-12-31"
    cfg.training.dataset.preload = True

    cfg.training.validation_dataset.start_date = "2020-01-01"
    cfg.training.validation_dataset.end_date = "2020-12-31"
    cfg.training.validation_dataset.preload = True

    cfg.training.scheduler.wsd.enabled = True
    cfg.training.scheduler.wsd.warmup = 50
    cfg.training.scheduler.wsd.decay = 0.2

    # Decide worker count from visible GPUs (or N_WORKERS)
    gpu_ids = _parse_visible_gpu_ids()
    n_workers = len(gpu_ids)
    total_trials = int(os.environ.get("TOTAL_TRIALS", "300"))
    trials_per_worker = math.ceil(total_trials / max(1, n_workers))

    logging.info("Launching %d workers over GPUs: %s", n_workers, gpu_ids)
    logging.info(
        "Each worker will run up to %d trials (total target: %d)",
        trials_per_worker,
        total_trials,
    )

    procs: list[Process] = []
    for i, gpu in enumerate(gpu_ids):
        p = Process(
            target=_worker, args=(cfg, gpu, trials_per_worker, study_name), daemon=False
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Summarize best params (any worker may have updated the study)
    study = optuna.load_study(study_name=study_name, storage=STORAGE_URL)
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")

    for trial in study.best_trials:
        print(f"  Values: {trial.values}")
        print("  Params:")
        for k, v in trial.params.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

