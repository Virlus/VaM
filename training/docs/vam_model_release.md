# 🔬 Model and inference API

**References:** \[[Training recipes](../README.md#initialize-with-offline-dataset)\] \[[Online DAgger/Learn from Experience](../../docs/online.md)\]

## Paper terminology

| Paper term | Meaning | Code / data name |
| :---: | --- | :---: |
| Action expert | Flow-matching policy that predicts action chunks | `action_expert` |
| Distributional value expert | Categorical predictor; its expectation is the scalar value | `value_expert`, `value_prediction` |
| Value cache | Past scalar predictions used as task-progress memory | History of `value_prediction` |
| Progress initialization | Gaussian labels centered at normalized demo progress `t/T` | `value_mc_learning` |
| Advantage condition | Binary action-quality label; inference requests the positive condition | `advantage`, advantage tokens |

BC is unconditioned flow matching during initialization. Warmup and online action
updates use **advantage-conditioned flow matching**; both use the code's `bc` loss
path, with conditioning controlled by the model and data configuration (paper Sec. 2.3).

## Model components

| `VAMConfig` field | Role |
| :---: | --- |
| `action_obs_encoder` | Encode action observations and cached state/value history |
| `action_expert` | Predict action chunks |
| `value_obs_encoder` | Encode observations for value estimation |
| `value_expert` | Predict a categorical value distribution over a fixed support |

| Setting | Behavior |
| :---: | --- |
| `observation_horizon` in each encoder | Independent action/value history lengths |
| Image encoder | SigLIP single-frame or video encoding |
| History tokens | GRU encoding of proprioception and the value cache |
| Progress initialization / TD | Gaussian progress labels / projected return distributions |
| Value target settings | Configurable support, bins, sigma, gamma and terminal masks |
| Joint delta transform | Relative joints with absolute gripper targets |

The cache uses **past** predictions: the current value estimate arrives asynchronously
with action generation (paper Sec. 3.2.1).

When loading a checkpoint, match branch horizons, image encoders, history tokens
and normalization to its training recipe. Gemma/PaliGemma, SigLIP, LoRA and history
parameters use the checkpoint's original names.

## Sampling and RTC

Ordinary and real-time chunking (RTC) inference use `VAM.sample_actions`:

```python
actions = model.sample_actions(rng, observation, enable_rtc=False)
actions = model.sample_actions(
    rng,
    observation,
    enable_rtc=True,
    prefix_actions=normalized_prefix,
    inference_delay=delay_steps,
    prefix_attention_horizon=remaining_steps,
    max_guidance_weight=10.0,
    prefix_attention_schedule="exp",
)
```

| Argument / behavior | Usage |
| :---: | --- |
| Ordinary sampling | Batched observations |
| RTC observation and prefix | One observation; normalized, unbatched full-horizon prefix |
| `inference_delay`, `prefix_attention_horizon` | Measured in action steps |
| Prefix width | Padded to the model action width |
| `noise`, `num_steps`, `cfg_guidance_weight` | Controls shared by both modes |
| Custom JIT wrappers | Keep `enable_rtc`, `cfg_guidance_weight`, `prefix_attention_schedule` static |
| Episode reset | First inference establishes the next prefix |

The supplied policy normalizes cached absolute targets and aligns the prefix with
elapsed time. Deployment RTC and human-prefix settings live in
[`deployment/configs/policy/vam.yaml`](../../deployment/configs/policy/vam.yaml).

## Streaming recipes

Recipes are defined in [`training/config`](../src/vam/training/config/__init__.py).
Use the [online launcher](../../docs/online.md#launch) for the split roles.

| Recipe | Purpose |
| :---: | :---: |
| `pi05_aloha_async_recap_dummy` | Shared async template |
| `pi05_aloha_async_recap` | Joint streaming trainer |
| `pi05_aloha_async_recap_split_action` | Action trainer |
| `pi05_aloha_async_recap_split_value` | Value trainer |
| `pi05_aloha_async_recap_streaming_validation` | Short streaming validation |
