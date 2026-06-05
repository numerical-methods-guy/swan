# Optuna Hyperparameter Tuning

SWAN includes an Optuna tuning script, `optuna_tune.py`, for hyperparameter search. The base experiment setup is read from `config_paradis.yaml`, while the hyperparameter search ranges are read from `optuna_search_spaces.yaml`.

Optuna does not overwrite `config_paradis.yaml`. For each trial, it makes a temporary in-memory copy of the config, inserts the sampled hyperparameters, trains the model, and returns validation loss to Optuna. Results are saved under `optuna_results/`.

## Basic usage

To tune SGD hyperparameters, run:

```bash
python optuna_tune.py --config config_paradis.yaml --optimizer sgd \
  --search_space optuna_search_spaces.yaml \
  --n_trials 40 \
  --study_name sgd_tuning
```

Replace `sgd` with another supported optimizer, such as:

```text
adam
adamw
muon
mud
gauss_newton
```

By default, Optuna tunes only the optimizer-specific section of `optuna_search_spaces.yaml`. For example, `--optimizer sgd` uses the `sgd:` section of the search-space file.

## Search spaces

The file `optuna_search_spaces.yaml` defines what Optuna is allowed to try. Each parameter has a `config_key`, which tells Optuna where to write the sampled value in the temporary trial config.

Example:

```yaml
sgd:
  learning_rate:
    config_key: training.sgd.learning_rate
    type: float
    low: 1.0e-4
    high: 1.0e-1
    log: true

  momentum:
    config_key: training.sgd.momentum
    type: float
    low: 0.0
    high: 0.99

  weight_decay:
    config_key: training.sgd.weight_decay
    type: categorical
    choices: [0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3]
```

This tells Optuna to sample SGD hyperparameters and write them into the corresponding `training.sgd.*` config fields for that trial.

To change tuning ranges, edit `optuna_search_spaces.yaml`. Do not edit `optuna_tune.py` just to change search ranges.

If you change the search space, use a new `--study_name` so results from different search spaces do not get mixed.

## Global parameters

The search-space file may also contain a `global:` section for model-level parameters such as:

```text
hidden_dim
num_layers
num_vels
bias_channels
```

These are not tuned unless `--tune_global` is passed.

To tune both SGD hyperparameters and global model parameters, run:

```bash
python optuna_tune.py --config config_paradis.yaml --optimizer sgd \
  --search_space optuna_search_spaces.yaml \
  --tune_global \
  --n_trials 40 \
  --study_name sgd_global_tuning
```

Use `--tune_global` only when the goal is to tune the full model/optimizer pipeline. For a controlled optimizer comparison, leave global model parameters fixed in `config_paradis.yaml` and omit `--tune_global`.

## Common flags

| Flag | Meaning |
|---|---|
| `--config` | Base config file, usually `config_paradis.yaml`. |
| `--optimizer` | Optimizer to tune. |
| `--search_space` | YAML file containing Optuna search ranges. |
| `--n_trials` | Number of Optuna trials. Each trial trains once with one sampled hyperparameter setting. |
| `--study_name` | Name for the Optuna study and output files. Use a new name when changing the search space or experiment setup. |
| `--tune_global` | Also tune the `global:` section of the search-space file. |
| `--output_dir` | Directory for Optuna outputs. Default: `optuna_results/`. |
| `--storage` | Optional Optuna storage backend, for example `sqlite:///optuna_sgd.db`. |

## Test run

Before a long tuning run, use a small test:

```bash
python optuna_tune.py --config config_paradis.yaml --optimizer sgd \
  --search_space optuna_search_spaces.yaml \
  --n_trials 3 \
  --study_name sgd_smoke_test \
  --training.pretrain_epochs 1 \
  --training.finetune_epochs 0 \
  --data.num_train_examples 2 \
  --data.num_val_examples 1 \
  --data.batch_size 1 \
  --data.nlat 8 \
  --data.nlon 16 \
  --model.paradis.hidden_dim 4 \
  --model.paradis.num_layers 1
```

This only checks that Optuna, the optimizer strategy, and the training pipeline run.

## Output

After tuning, results are written to:

```text
optuna_results/
```

The main files are:

```text
<study_name>_trials.csv
<study_name>_best_params.yaml
```

The CSV file contains all trials. The YAML file records the best validation loss and best hyperparameters.