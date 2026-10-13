# 🦾 Robot deployment

**References:** \[[Installation](../README.md#installation)\] \[[Training](../training/README.md)\] \[[Online DAgger/Learn from Experience](../docs/online.md)\]

## 🚀 Quick start

1. Install the [robot environment](../README.md#installation) with `--extra hardware`.
2. Fill in your camera serials, CAN channels and robot poses in [Configuration](#configuration).
3. Start [Data collection](#data-collection) to record demonstrations, or [Offline DAgger warmup data collection](#offline-dagger-warmup-data-collection) with an initialized checkpoint.

Run the following from the repository root:

```bash
cd deployment
mkdir -p configs/local
```

All commands below run from **`deployment/`**. Use the same local config in each terminal.

## Configuration

| Config | Use |
| :---: | --- |
| [`configs/collect.yaml`](configs/collect.yaml) | Data collection |
| [`configs/rollout.yaml`](configs/rollout.yaml) | Offline DAgger warmup data collection |
| [`configs/example.yaml`](configs/example.yaml) | Online DAgger/Learn from Experience |

| Setting | Where to edit |
| :---: | --- |
| Task, run name, recording directory | `task`, `run_name`, `recording_output_dir` |
| Three camera serials | `camera.ego_serial`, `camera.left_serial`, `camera.right_serial` |
| CAN channels and RPC ports | Robot entries in [`configs/devices/i2rt_dual_bilateral.yaml`](configs/devices/i2rt_dual_bilateral.yaml) |
| Home poses, takeover alignment, random reset | `robot` mapping |
| Initial model | `checkpoint`; match the recipe in `policy.params.pi_config_name` to its architecture |
| Python interpreters | `runtime.python` |
| Upload destination | `remote` mapping and `upload.enabled` |

Create `configs/local/collect.yaml` for machine-specific settings:

```yaml
# @package _global_
defaults:
  - /collect
  - _self_
task: fold the t-shirt
run_name: example_run
camera:
  ego_serial: YOUR_EGO_SERIAL
  left_serial: YOUR_LEFT_SERIAL
  right_serial: YOUR_RIGHT_SERIAL
```

Local overrides in `configs/local/` are ignored by Git.

<a id="collection"></a>

## Data collection

Collect successful expert demonstrations (`D_demo`) for action-expert behavior cloning
and value-expert progress initialization.

Start each component in a separate terminal:

```bash
# Terminal 1: robot and cameras
../.venv/bin/python run.py --config-name local/collect component=devices

# Terminal 2: demonstration recording
../.venv/bin/python run.py --config-name local/collect component=collector
```

| Control / output | Behavior |
| :---: | --- |
| Teaching-handle button | Start / finish recording |
| `D` | Discard the attempt; published episode numbers stay consecutive |
| `data/collect/<run_name>/ready/` | Completed upload packages |
| `data/collect/<run_name>/datasets/` | Accumulated action/value training repositories |

To upload, configure `remote` and run:

```bash
../.venv/bin/python run.py --config-name local/collect component=upload upload.enabled=true
```

**Result:** paired action/value LeRobot datasets of successful demonstrations.
Allow background saves to finish, then continue with
[Initialize with offline dataset](../training/README.md#initialize-with-offline-dataset).

<a id="fixed-policy-rollout"></a>

## Offline DAgger warmup data collection

This is the collection part of the paper’s **offline RL warmup**. Deploy the initialized
action/value pair and record policy-controlled segments, human interventions (HI),
and terminal success/failure outcomes.

Create `configs/local/rollout.yaml`, replacing the checkpoint path and camera serials.
Use the checkpoint's original normalization and matching model history settings.

```yaml
# @package _global_
defaults:
  - /rollout
  - _self_
task: fold the t-shirt
run_name: example_run
checkpoint: /path/to/initialized/checkpoint
camera:
  ego_serial: YOUR_EGO_SERIAL
  left_serial: YOUR_LEFT_SERIAL
  right_serial: YOUR_RIGHT_SERIAL
```

Apply the same robot settings as for collection. Start each component in a separate terminal:

```bash
# Terminal 1: robot and cameras
../.venv/bin/python run.py --config-name local/rollout component=devices

# Terminal 2: policy rollouts and human intervention
../.venv/bin/python run.py --config-name local/rollout component=policy
```

| Control / output | Behavior |
| :---: | --- |
| Leader takeover | Record human-intervention (HI) segments and takeover boundaries |
| `S` / `F` | Mark success / failure |
| `D` | Discard the episode and reuse its index |
| `data/rollout/<run_name>/ready/` | Warmup rollouts stored as paired action/value LeRobot packages |

**Result:** warmup rollout datasets with human interventions and episode outcomes.
Continue with [Offline DAgger warmup](../training/README.md#offline-dagger-warmup), then
[Online DAgger/Learn from Experience](../docs/online.md).
