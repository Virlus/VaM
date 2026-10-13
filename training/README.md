# 🧠 Initialization and DAgger warmup

**References:** \[[Installation](../README.md#installation)\] \[[Online DAgger/Learn from Experience](../docs/online.md)\] \[[Model and RTC](docs/vam_model_release.md)\]

Run commands in this guide from **`training/`**, using `../.venv/bin/python`.

<a id="offline-training"></a>

## Initialize with offline dataset

Paper stage: **offline demonstration initialization** (Sec. 3.3.1). Successful expert
demonstrations train the action expert with behavior cloning (BC) and initialize the
distributional value expert with normalized progress labels centered at `t/T`.

Choose a base recipe in [`src/vam/training/config/__init__.py`](src/vam/training/config/__init__.py):

| Recipe | Action / value history | Objective |
| :---: | :---: | --- |
| `pi05_aloha_value_expert` | 1 / 1 frames | BC + progress initialization |
| `pi05_aloha_history_value_expert` | 4 / 2 frames | BC + progress initialization, with value caching |

The implementation names the progress-initialization loss `value_mc_learning`.

Use the [collected dataset paths](preprocess_data/README.md#use-collected-demonstrations)
or an existing LeRobot dataset. Configure the initial model and these inputs:

| Input | Config field / CLI flag |
| :---: | --- |
| Dataset | `data.repo_id` / `--data.repo-id` |
| Initial parameters | `weight_loader.params_path` / `--weight-loader.params-path` |
| Original normalization | `data.assets.norm_stats_path` / `--data.assets.norm-stats-path` |
| Asset identity | `data.assets.asset_id` / `--data.assets.asset-id` |
| Experiment name | `exp_name` / `--exp-name` |

Example launch (replace dataset and path values):

```bash
../.venv/bin/python scripts/train.py pi05_aloha_value_expert \
  --exp-name example_run \
  --data.repo-id example/offline \
  --data.assets.asset-id example/offline \
  --data.assets.norm-stats-path /path/to/norm_stats.json \
  --weight-loader.params-path /path/to/base/checkpoint/params
```

- Use `HF_LEROBOT_HOME` for your local dataset root and configure W&B credentials in the shell.
- For history training, prepare `value_prediction` labels with the [annotation tools](preprocess_data/README.md#annotation).
- Tune batch size, steps and branch horizons through the recipe or CLI; append `--help` for available flags.

**Output:** the initialized action/value pair, saved under
`checkpoints/<recipe>/<exp_name>/<step>/` with inference `params/` and normalization `assets/`.

## Normalization

To compute statistics for a new dataset, first set its repo ID and asset ID in the recipe:

```bash
../.venv/bin/python scripts/compute_norm_stats_async.py --config-name pi05_aloha_value_expert
```

This writes single-frame state/action statistics under `assets/<config_name>/<asset_id>/norm_stats.json`.
For continued training, use the initial checkpoint's original statistics.

<a id="offline-recap"></a>

## Offline DAgger warmup

Paper stage: **offline RL warmup** (Sec. 3.3.2). Adapt the value expert with TD learning,
then improve the action expert with advantage-conditioned flow matching.

Set the dataset and initialization fields in the selected recipe. For `run_recap.py`,
set `model.action_expert.pi06=False` in that base recipe; the launcher enables
advantage conditioning for the action-update step. Then run:

```bash
../.venv/bin/python scripts/run_recap.py pi05_aloha_history_value_expert --exp-name example_run
```

| Operation | Result |
| :---: | --- |
| 1. TD value learning | TD-trained value expert |
| 2. Value annotation | `value_prediction` from the warmup value expert |
| 3. Advantage annotation | Value-derived advantages and binary advantage conditions |
| 4. Advantage-conditioned flow matching | Advantage-conditioned policy |

Use `--value-repo-id` and `--action-repo-id` for separate role datasets;
`--offline-repo-id` selects the normalization source. Append `--help` for stage settings.

**Output:** a **TD-trained value expert** and an **advantage-conditioned policy**
for online learning. The launcher saves value and action stages under `checkpoints/<recipe>/value_<exp_name>/`
and `checkpoints/<recipe>/action_<exp_name>/`.

## Reference

| Topic | Guide |
| --- | :---: |
| Online ingestion, split training and resume | [Online DAgger/Learn from Experience](../docs/online.md) |
| Model branches, checkpoint settings and RTC | [Model API](docs/vam_model_release.md) |
| Dataset registration and annotation | [Data tools](preprocess_data/README.md) |
