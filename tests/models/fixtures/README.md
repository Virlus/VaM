# Sampling reference

| Item | Setting |
| --- | --- |
| File | `sampling_before_unification.npz` |
| Source commit | `0028766c`, before ordinary and RTC sampling were combined |
| Model | Synthetic CPU `small_model` from [`test_vam_release.py`](../test_vam_release.py) |
| Random keys | Initialization 0; sampling 5 |
| Precision / sampling | float32; three denoising steps |
| History / CFG | History 1 or 2; guidance 1 or 2 |
| Modes | Ordinary; RTC `ones`, `zeros`, `linear`, `exp` |
| Advantage token | `model.pi06=True`; one valid token with ID 1 |
| RTC prefix | `arange(6).reshape(2, 3) / 5` |
| RTC timing | Delay 1, attention horizon 1, maximum guidance 2 |
| NPZ keys | `<history>/<cfg_weight>/<mode>` |

Historical generation called `sample_actions` and `sample_actions_rtc` separately.
[`test_sampling.py`](../test_sampling.py) uses `1e-6` absolute/relative tolerance.

| Recorded comparison | Maximum difference |
| --- | --- |
| Ordinary, local CPU | Exact |
| RTC differentiation refactor | `2.98e-8` |
| Ordinary, cloud CPU | `2.39e-7` |
