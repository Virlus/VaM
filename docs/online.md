# 🔄 Online DAgger/Learn from Experience

**References:** \[[Installation](../README.md#installation)\] \[[Robot setup](../deployment/README.md#configuration)\] \[[Initialize with offline dataset](../training/README.md#initialize-with-offline-dataset)\]

This guide covers two paper stages (Sec. 3.3.2). Online DAgger starts from the
warmup outputs: a **TD-trained value expert** and an **advantage-conditioned policy**.
Learn from Experience (LfE) continues from the latest Online DAgger pair.

| Phase | New rollout data | Human-intervention replay |
| :---: | --- | --- |
| Online DAgger | Policy rollouts with human interventions (HI) | Grows as new HI segments arrive |
| Learn from Experience (LfE) | Autonomous rollouts with sparse terminal outcomes | Retains the interventions collected before LfE |

For LfE, run episodes autonomously and mark their terminal outcome with `S` / `F`.

| Role | Work |
| :---: | --- |
| Robot: `devices`, `policy`, `upload` | Run episodes, record corrections and transfer completed packages |
| Training: `ingest` | Register rollouts, threshold advantages and publish paired dataset views |
| Training: `action`, `value` | Advantage-conditioned flow matching / TD value learning; publish the updated pair |

## Configuration

Use the **warmup checkpoint** containing the TD-trained value expert and
advantage-conditioned policy. Create these private
overlays after completing the [robot settings](../deployment/README.md#configuration).
Replace the path and host placeholders with your values.

**`deployment/configs/local/example.yaml`:**

```yaml
# @package _global_
defaults:
  - /example
  - _self_
run_name: example_run
checkpoint: /path/to/initial/checkpoint
remote:
  root: /path/to/async
  host: YOUR_TRAINING_HOST
  username: YOUR_USERNAME
```

Add the robot and camera overrides from your deployment setup to this overlay.

**`training/configs/local/example.yaml`:**

```yaml
include: ../example.yaml
variables:
  run_name: example_run
  split_root: /path/to/async
job:
  offline_dataset_home: /path/to/offline/lerobot
  base_params: /path/to/initial/checkpoint/params
  norm_assets_dir: /path/to/initial/checkpoint/assets
  norm_stats_path: /path/to/initial/checkpoint/assets/example/offline/norm_stats.json
```

| Match across hosts | Deployment | Training |
| :---: | :---: | --- |
| Run identity | `run_name` | `variables.run_name` |
| Shared storage | `remote.root` | `variables.split_root` |
| Initial model | `checkpoint` | `job.base_params`, `job.norm_assets_dir`, `job.norm_stats_path` |

| Training setting | Configure in the training overlay |
| :---: | --- |
| Offline action/value datasets | `variables.action_offline_repo`, `variables.value_offline_repo`, relative to `job.offline_dataset_home` |
| GPU allocation | `job.action_devices`, `job.value_devices`; separate GPU sets when sharing a host |
| Round length / total steps | `job.round_steps`, `job.num_train_steps` |
| Batch size / FSDP / seed | `job.trainer_overrides.action`, `.value`, `.common` |

Use the same code revision and shared storage on training hosts. A typical setup runs
`ingest`/`value` on one host and `action` on another. Configure W&B credentials in each training shell.

## Launch

1. Start the assigned training roles in separate terminals, from the **repository root**:

```bash
.venv/bin/python training/run_online.py training/configs/local/example.yaml --component ingest
.venv/bin/python training/run_online.py training/configs/local/example.yaml --component action
.venv/bin/python training/run_online.py training/configs/local/example.yaml --component value
```

2. On the robot host, start each component in a separate terminal from **`deployment/`**:

```bash
../.venv/bin/python run.py --config-name local/example component=devices
../.venv/bin/python run.py --config-name local/example component=upload
../.venv/bin/python run.py --config-name local/example component=policy
```

Use the [rollout controls](../deployment/README.md#offline-dagger-warmup-data-collection) during collection.
The deployed value snapshot supplies the value cache and current-rollout advantage
labels. Action and value updates run in parallel; each merged checkpoint contains both
updated experts and is loaded between episodes.

## Replay and results

| Expert | Batch sources | Default mixing |
| :---: | --- | :---: |
| Action | Expert demonstrations, historical HI segments, annotated current rollouts | 1 : 1 : 1 |
| Value | Replay (demonstrations + earlier rollouts), current rollout transitions | 1 : 1 |

**Each round produces:** an updated action/value pair and new rollouts for replay.
Online DAgger also expands the HI buffer; LfE keeps that buffer fixed. The final result
is the action/value pair after the last LfE round (Algorithm 1).

## Resume

| Task | Setting |
| :---: | --- |
| Continue an existing run | Keep run name, paths, base checkpoint and normalization; set `job.resume: true` |
| Start an independent run | Choose a new run name and output root; set `job.resume: false` |
| Interpret checkpoint status | `policy_id` is the policy generation; `checkpoint_index` identifies a checkpoint step |

Split training restores published weights/steps and reinitializes optimizer/EMA state.
The launcher reads resume mode from YAML.
