# Rollout frame reference

| File / source | Contents |
| --- | --- |
| `source.npz` | Synthetic RGB and absolute nanosecond timestamps |
| `source.json` | Recipe and explicit episode outcome |
| Role NPZ files | Frozen action/value outputs from 2026-09-18 |
| Generator | Historical `NEDF2AlohaConverter.convert_episode()` and `_compute_advantages_for_table()` |
| Converter SHA-256 | `664a20e0277044fef8df26459c0718fcd56bbfe79e448ab44f23ae53f4b98e9b` |

Generation used the deployed fold-tshirt configuration on `1019_wenye`, replacing
the writer with an array sink. It retained JPEG-95 roundtrips, timestamp normalization
and latency compensation. Historical evidence: `/tmp/lerobot-restore.fhlZzN/final_oracle/oracle.py`
and `oracle.log` on that host.

| Case | Coverage |
| --- | --- |
| `success`, `failure` | Episode outcomes |
| `takeover`, `compacted_takeover` | Human-control boundaries and half-open timeline compaction |
| `gaps`, `sparse_values` | State/action gaps and sparse value predictions |
| `epoch_rgb` | Textured RGB, epoch-sized timestamps and camera latency |

[Parity tests](../../utils/direct_collection/test_legacy_parity.py) compare exact dtype,
shape and every value, including float64 raw GAE. Small 16 × 16 writer/SHM tests use
H.264; recording defaults use AV1, yuv420p, `g=2`, `crf=30`.
