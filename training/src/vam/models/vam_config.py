import dataclasses
from typing import TYPE_CHECKING, Literal, TypeAlias

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from vam.models import model as _model
import vam.models.gemma as _gemma
from vam.shared import array_typing as at
import vam.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from vam.models.vam import VAM

ImageEncoderType = Literal["siglip", "siglip_video"]
HistoryTokenStrategy = Literal["gru"]
ObsEncoderKind: TypeAlias = Literal["paligemma"]
ActionExpertKind: TypeAlias = Literal["paligemma"]
PrefixAttentionSchedule = Literal["ones", "zeros", "linear", "exp"]


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ImageEncoderConfig:
    """Config for the image encoder (single-frame or multi-frame video)."""

    encoder_type: ImageEncoderType = "siglip"
    temporal_attention_interval: int = 4  # only used when encoder_type == "siglip_video"

    def __post_init__(self):
        if self.encoder_type not in ("siglip", "siglip_video"):
            raise ValueError(f"Invalid encoder_type={self.encoder_type}")
        if self.temporal_attention_interval < 1:
            raise ValueError(f"temporal_attention_interval must be >= 1, got {self.temporal_attention_interval}")


@dataclasses.dataclass(frozen=True)
class HistoryTokenConfig:
    """Config for GRU-based history token injection into the action prefix."""

    use_state_tokens: bool = False
    use_value_tokens: bool = False
    strategy: HistoryTokenStrategy = "gru"

    def __post_init__(self):
        if self.strategy != "gru":
            raise ValueError(f"Invalid history token strategy={self.strategy}")


@dataclasses.dataclass(frozen=True)
class ObsEncoderConfig:
    """Config for an observation encoder component."""

    kind: ObsEncoderKind = "paligemma"
    variant: str = "gemma_2b"
    observation_horizon: int = 1
    image_encoder: ImageEncoderConfig = dataclasses.field(default_factory=ImageEncoderConfig)
    history_tokens: HistoryTokenConfig = dataclasses.field(default_factory=HistoryTokenConfig)

    def __post_init__(self):
        if self.kind != "paligemma":
            raise ValueError(f"Invalid obs encoder kind={self.kind}")
        if self.observation_horizon < 1:
            raise ValueError(f"observation_horizon must be >= 1, got {self.observation_horizon}")


@dataclasses.dataclass(frozen=True)
class PaliGemmaConfig(ObsEncoderConfig):
    """Backward-compatible alias for the default PaliGemma observation encoder."""

    kind: ObsEncoderKind = "paligemma"


@dataclasses.dataclass(frozen=True)
class ActionExpertConfig:
    """Config for the action expert (flow-matching diffusion decoder)."""

    kind: ActionExpertKind = "paligemma"
    variant: _gemma.Variant = "gemma_300m"
    pi05: bool = False
    pi06: bool = False
    discrete_state_input: bool = None  # type: ignore  # defaults to pi05 in __post_init__
    advantage_conditioning: bool = None  # type: ignore  # defaults to pi06 in __post_init__
    async_action_horizon: int = -1

    def __post_init__(self):
        if self.kind != "paligemma":
            raise ValueError(f"Invalid action expert kind={self.kind}")
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.advantage_conditioning is None:
            object.__setattr__(self, "advantage_conditioning", self.pi06)


@dataclasses.dataclass(frozen=True)
class ValueExpertConfig:
    """Categorical value head: HL-Gauss targets for MC, projected distributions for TD."""

    # Value head parameters
    n_bins: int = 256
    vmin: float = 0.0
    vmax: float = 1.0
    gaussian_sigma: float = 3.0 / 512

    def __post_init__(self):
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2 for discrete value heads, got {self.n_bins}")
        if self.vmin >= self.vmax:
            raise ValueError("vmin must be less than vmax")
        if self.gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")


# ---------------------------------------------------------------------------
# Top-level policy configs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class VAMConfig(_model.BaseModelConfig):
    action_obs_encoder: ObsEncoderConfig = dataclasses.field(default_factory=ObsEncoderConfig)
    action_expert: ActionExpertConfig | None = dataclasses.field(default_factory=ActionExpertConfig)
    value_obs_encoder: ObsEncoderConfig = dataclasses.field(
        default_factory=lambda: ObsEncoderConfig(variant="gemma_300m")
    )
    value_expert: ValueExpertConfig | None = dataclasses.field(default_factory=ValueExpertConfig)
    # TD learning params (shared between online and target value network)
    td_learning: bool = False
    gamma: float = 0.99
    target_update_interval: int = 1
    target_update_tau: float = 1.0
    # Pi06-specific
    advantage_threshold: float = 0.7
    dtype: str = "bfloat16"
    # BaseModelConfig fields with policy-specific defaults
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore

    def __post_init__(self):
        if self.action_expert is None and self.value_expert is None:
            raise ValueError("At least one expert branch must be enabled")
        if self.td_learning and self.value_expert is None:
            raise ValueError("td_learning requires value_expert")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if self.target_update_interval < 1 or not 0 < self.target_update_tau <= 1:
            raise ValueError("Target update interval must be positive and tau must be in (0, 1]")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)

    # ------------------------------------------------------------------
    # Branch presence flags
    # ------------------------------------------------------------------

    @property
    def has_action_expert_branch(self) -> bool:
        return self.action_expert is not None

    @property
    def has_value_expert_branch(self) -> bool:
        return self.value_expert is not None

    # ------------------------------------------------------------------
    # Derived properties used by transforms, policies, and weight loaders
    # ------------------------------------------------------------------

    @property
    def paligemma_variant(self) -> _gemma.Variant:
        return self.action_obs_encoder.variant  # type: ignore[return-value]

    @property
    def action_expert_variant(self) -> _gemma.Variant:
        return self.action_expert.variant if self.action_expert else "gemma_300m"

    @property
    def value_variant(self) -> _gemma.Variant:
        return self.value_obs_encoder.variant if self.value_obs_encoder else "gemma_300m"  # type: ignore[return-value]

    @property
    def observation_horizon(self) -> int:
        horizons = [self.action_obs_encoder.observation_horizon]
        if self.value_obs_encoder is not None:
            horizons.append(self.value_obs_encoder.observation_horizon)
        return max(horizons)

    @property
    def action_observation_horizon(self) -> int:
        return self.action_obs_encoder.observation_horizon

    @property
    def value_observation_horizon(self) -> int:
        return self.value_obs_encoder.observation_horizon if self.value_obs_encoder else 1

    @property
    def pi05(self) -> bool:
        return self.action_expert.pi05 if self.action_expert else False

    @property
    def pi06(self) -> bool:
        return self.action_expert.pi06 if self.action_expert else False

    @property
    def discrete_state_input(self) -> bool:
        return self.action_expert.discrete_state_input if self.action_expert else False

    @property
    def advantage_conditioning(self) -> bool:
        return self.action_expert.advantage_conditioning if self.action_expert else False

    @property
    def async_action_horizon(self) -> int:
        return self.action_expert.async_action_horizon if self.action_expert else -1

    @property
    def use_history_state_tokens(self) -> bool:
        return self.action_obs_encoder.history_tokens.use_state_tokens

    @property
    def use_history_value_tokens(self) -> bool:
        return self.action_obs_encoder.history_tokens.use_value_tokens

    @property
    def action_image_encoder_type(self) -> ImageEncoderType:
        return self.action_obs_encoder.image_encoder.encoder_type

    @property
    def value_image_encoder_type(self) -> ImageEncoderType:
        return self.value_obs_encoder.image_encoder.encoder_type if self.value_obs_encoder else "siglip"

    @property
    def temporal_attention_interval(self) -> int:
        return self.action_obs_encoder.image_encoder.temporal_attention_interval

    @property
    def n_bins(self) -> int:
        return self.value_expert.n_bins if self.value_expert else 256

    @property
    def discrete_value(self) -> bool:
        return True

    @property
    def vmin(self) -> float:
        return self.value_expert.vmin if self.value_expert else -1.0

    @property
    def vmax(self) -> float:
        return self.value_expert.vmax if self.value_expert else 1.0

    @property
    def gaussian_sigma(self) -> float:
        return self.value_expert.gaussian_sigma if self.value_expert else 3.0 / 512

    @property
    def value_loss(self) -> Literal["hl_gauss", "distributional"]:
        return "distributional" if self.td_learning else "hl_gauss"

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi06:
            return _model.ModelType.PI06
        if self.pi05:
            return _model.ModelType.PI05_VALUE_EXPERT
        return _model.ModelType.PI0_VALUE_EXPERT

    @override
    def fake_obs(self, batch_size: int = 1) -> _model.Observation:
        observation_spec, _, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    @override
    def fake_act(self, batch_size: int = 1) -> _model.Actions:
        _, action_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)

    @override
    def create(self, rng: at.KeyArrayLike) -> "VAM":
        from vam.models.vam import VAM

        return VAM(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions, _model.Values]:
        if self.observation_horizon == 1:
            image_shape = [batch_size, *_model.IMAGE_RESOLUTION, 3]
            image_mask_shape = [batch_size]
            state_shape = [batch_size, self.action_dim]
        else:
            image_shape = [batch_size, self.observation_horizon, *_model.IMAGE_RESOLUTION, 3]
            image_mask_shape = [batch_size, self.observation_horizon]
            state_shape = [batch_size, self.observation_horizon, self.action_dim]
        image_spec = jax.ShapeDtypeStruct(image_shape, jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct(image_mask_shape, jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct(state_shape, jnp.float32),
                state_mask=jax.ShapeDtypeStruct(image_mask_shape, jnp.bool_),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                tokenized_value_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_value_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                value_prediction=(
                    jax.ShapeDtypeStruct([batch_size, self.observation_horizon, 1], jnp.float32)
                    if self.use_history_value_tokens
                    else None
                ),
                value_prediction_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.observation_horizon], bool)
                    if self.use_history_value_tokens
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        value_spec = (
            jax.ShapeDtypeStruct([batch_size, 1], jnp.float32)
            if not self.discrete_value
            else jax.ShapeDtypeStruct([batch_size, self.n_bins], jnp.float32)
        )

        return observation_spec, action_spec, value_spec

    def get_freeze_filter(self, freeze_value: bool = False) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        has_lora = False
        base_filters = []
        exclusions = []
        gemma_params_filter = [nnx_utils.PathRegex("PaliGemma.*")]
        action_expert_params_filters = [
            nnx_utils.PathRegex("PaliGemma.*_1.*"),
            nnx_utils.PathRegex("action_in_proj/.*"),
            nnx_utils.PathRegex("state_proj/.*"),
            nnx_utils.PathRegex("time_mlp_in/.*"),
            nnx_utils.PathRegex("time_mlp_out/.*"),
            nnx_utils.PathRegex("action_out_proj/.*"),
        ]
        value_llm_filter = nnx_utils.PathRegex("ValuePaliGemma/llm/.*")
        value_img_filter = nnx_utils.PathRegex("ValuePaliGemma/img/.*")
        value_head_filters = [
            nnx_utils.PathRegex("value_text_proj/.*"),
            nnx_utils.PathRegex("value_out_proj/.*"),
            nnx_utils.PathRegex("value_params.*"),
        ]
        target_value_filters = [
            nnx_utils.PathRegex("TargetValuePaliGemma/.*"),
            nnx_utils.PathRegex("target_value_text_proj/.*"),
            nnx_utils.PathRegex("target_value_out_proj/.*"),
            nnx_utils.PathRegex("target_value_params.*"),
        ]
        if freeze_value:
            base_filters.append(value_img_filter)
        base_filters.extend(target_value_filters)

        has_action_branch = self.has_action_expert_branch
        has_value_branch = self.has_value_expert_branch

        if has_action_branch and "lora" in self.paligemma_variant:
            base_filters.extend(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                exclusions.extend(nnx.Not(filter) for filter in action_expert_params_filters)
            if has_value_branch and "lora" not in self.value_variant and not freeze_value:
                exclusions.extend(nnx.Not(filter) for filter in value_head_filters)
                exclusions.append(nnx.Not(value_llm_filter))
                exclusions.append(nnx.Not(value_img_filter))
            has_lora = True
        elif has_action_branch and "lora" in self.action_expert_variant:
            base_filters.extend(action_expert_params_filters)
            if has_value_branch and "lora" not in self.value_variant and not freeze_value:
                exclusions.extend(nnx.Not(filter) for filter in value_head_filters)
                exclusions.append(nnx.Not(value_llm_filter))
                exclusions.append(nnx.Not(value_img_filter))
            has_lora = True
        elif has_value_branch and "lora" in self.value_variant and not freeze_value:
            base_filters.extend(value_head_filters)
            base_filters.append(value_llm_filter)
            base_filters.append(value_img_filter)
            has_lora = True

        if has_lora:
            exclusions.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))

        if not base_filters:
            return nnx.Nothing
        freeze_filter = nnx.Any(*base_filters)
        if exclusions:
            return nnx.All(freeze_filter, *exclusions)
        return freeze_filter
