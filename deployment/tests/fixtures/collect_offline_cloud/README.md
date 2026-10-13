# Offline cloud reference

| File | Contents |
| --- | --- |
| `source_*.npz` | RGB images, float32 joints and absolute-second `ts_<stream>` timestamps |
| `config.json` | Alignment, features, task and window settings; camera latency 33 ms |
| `reference.json` | Converter SHA-256, OpenCV version, source hashes and expected frame counts |
| Expected arrays | Historical converter output after a JPEG-95 roundtrip, before video encoding |

Generated on 2026-09-20 in the historical training environment. The exporter replaced
the source reader and dataset sink, retaining the converter's own alignment logic.

| Case group | Coverage |
| --- | --- |
| Timing | Short/padded chunks, regular and irregular camera cadence |
| Missing inputs | Unequal cameras, duplicate timestamps, shorter state stream |
| Images | 16 × 16 RGB; `rgb64` uses 64 × 64 textures for AV1 parity |
| Role selection | Action stride 1; value stride 30 plus the final aligned observation |

[Test implementation](../../consumer/test_lerobot_offline_cloud_parity.py)
