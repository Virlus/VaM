# Collection frame reference

| Item | Reference |
| --- | --- |
| File | `collect_offline_reference.npz` |
| Generator | Historical `third_party/openpi/preprocess_data/nedf2_to_lerobot_aloha.py`, `convert_episode` |
| Source checkout | `9f8e2f32d1f65da4742536d3a2cc0e38d2bfba7f` |
| Source SHA-256 | `f30b4a54c6fa5529010dd58db595d2d5fcfb5f23f5c052ffa20ff7337b054c05` |
| Synthetic input | [`make_streams`](../consumer/collect_samples.py): 12, 80 or 240 robot samples |
| Timing | Robot 100 Hz; camera 30 Hz; latency compensation 33 ms |
| Images | 16 × 16 RGB arrays |
| Recipe | Window 100; successful offline binary rewards; stride 1 |
| Keys | `<robot_sample_count>/<feature>`; 15 numeric/image fields including terminal labels |

Generation replaced the converter's raw-file loader with timestamped arrays and
captured each `add_frame` call. Relative timestamps and camera latency were applied
before conversion; RGB arrays passed directly to the frame builder.

| Check | Expected behavior |
| --- | --- |
| [Frame parity](../consumer/test_lerobot_frames.py) | Compare saved arrays against the shared builder |
| Value selection | Rows 0, 30, 60, … and the final row |
| LeRobot output | Regenerated indices/FPS, Parquet and six video streams |
