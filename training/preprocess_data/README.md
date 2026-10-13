# 🗂️ Data tools

**References:** \[[Data collection](../../deployment/README.md#data-collection)\] \[[Initialize with offline dataset](../README.md#initialize-with-offline-dataset)\] \[[Online DAgger/Learn from Experience](../../docs/online.md)\]

## Data flow

| Stage | Input | Output |
| :---: | --- | --- |
| Record demonstrations | Successful expert trajectories | `D_demo`, stored in paired LeRobot datasets |
| Record rollouts | Policy actions, HI segments, outcomes and value predictions | Warmup/current rollout datasets |
| Register | Completed uploaded packages | Training repositories and paired dataset views |
| Annotate values | Dataset + value checkpoint | `value_prediction` |
| Annotate advantages | Values, rewards and terminal labels | Raw advantages + binary advantage conditions |

Paired action/value datasets are storage views for the two experts. The paper’s data
sources are expert demonstrations, off-policy rollouts (including historical HI),
and current on-policy rollouts (Appendix E.3).

The online `ingest` role handles registration. Its paths and role datasets are configured
in the [online job](../../docs/online.md#configuration).

## Use collected demonstrations

The collector assembles complete repositories under `recording_output_dir/datasets/`.
With the default `run_name: example_run`:

| Training input | Local value |
| :---: | --- |
| `HF_LEROBOT_HOME` | Absolute path to `deployment/data/collect/example_run/datasets` |
| Action repo ID | `example_run` |
| Value repo ID | `example_run_value` |

After upload, the same repo IDs live under `<remote.root>/lerobot/example_run/`.
Use that directory as the training dataset root, or as `job.offline_dataset_home`
with these IDs in the [online configuration](../../docs/online.md#configuration).
Stop recording and finish pending saves/uploads before using the dataset for training.

## Annotation

Set the dataset and model fields in your [training recipe](../README.md#initialize-with-offline-dataset).
Run from **`training/`**:

```bash
../.venv/bin/python preprocess_data/annotate.py value pi05_aloha_value_expert \
  --ckpt-path /path/to/value/checkpoint/params
../.venv/bin/python preprocess_data/annotate.py advantage pi05_aloha_value_expert \
  --advantage-source value_prediction --advantage-method gae --advantage-obs-horizon 1
```

| Option | Effect |
| :---: | --- |
| `--percentile` | Threshold for the binary advantage condition; default 70 |
| `--inference-batch-size`, `--fsdp-devices` | Value inference resources |
| `--overwrite` | Replace the existing annotation column |
| `--help` | Show the selected command's full options |

## Data contracts

| Field / operation | Preserved behavior |
| :---: | --- |
| Frame construction | Shared [`aloha_frame_builder.py`](utils/aloha_frame_builder.py) aligns cameras, actions and terminal samples |
| Action/value strides | Applied when recording; ingestion keeps stored arrays and video bytes |
| Human interventions (HI) | Retain takeover boundaries and positive advantage conditions |
| `advantage_raw` | Dense-camera GAE retained through upload; online p70 labels are refreshed per policy |
| Value cache | Uses a history of recorded or model-annotated `value_prediction` |
| Publication | Action/value views published together with immutable advantage snapshots |
| Restart | Per-policy append/retry and replay cursors retained |
| Offline inputs | Linked read-only into the run's dataset home |
