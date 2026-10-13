import dataclasses
import logging
import random
from typing import Set, Union

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from termcolor import cprint
from typing_extensions import override

from vam.models import model as _model
from vam.models import vam_config
import vam.models.gemma as _gemma
from vam.models.gru import HistoryGRU
from vam.models.utils import value_distribution as _value_distribution
from vam.models.utils.attention import make_attn_mask
from vam.models.utils.history import _slice_image_history
from vam.models.utils.history import _slice_observation_history
from vam.models.utils.history import current_history_mask
from vam.models.utils.history import history_gru_token
from vam.models.utils.position_embedding import posemb_sincos
from vam.models.utils.rtc import get_prefix_weights
from vam.models.utils.rtc import guide_velocity
from vam.models.utils.rtc import rtc_velocity
from vam.models.utils.vision import _embed_image_tokens
from vam.models.utils.vision import _image_token_mask
from vam.models.utils.vision import _make_image_encoder
from vam.models.utils.vision import _siglip_init_image
from vam.models.utils.vision import _state_tokens
from vam.models.vam_config import PrefixAttentionSchedule
from vam.shared import array_typing as at

logger = logging.getLogger("vam")


class VAM(_model.BaseModel):
    """Joint action/value model with independently configurable observation encoders."""

    def __init__(self, config: vam_config.VAMConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.pi06 = config.pi06
        assert not self.pi06 or self.pi05, "Pi06 implies Pi05 is also enabled"

        self.discrete_value = config.discrete_value
        self.td_learning = config.td_learning
        self.gamma = config.gamma
        self.advantage_threshold = config.advantage_threshold
        self.action_observation_horizon = config.action_observation_horizon
        self.value_observation_horizon = config.value_observation_horizon
        self.action_image_encoder_type = config.action_image_encoder_type
        self.value_image_encoder_type = config.value_image_encoder_type
        self.use_history_state_tokens = config.use_history_state_tokens
        self.use_history_value_tokens = config.use_history_value_tokens
        self.vmin = config.vmin
        self.vmax = config.vmax
        self.gaussian_sigma = config.gaussian_sigma
        self.target_update_interval = config.target_update_interval
        self.target_update_tau = config.target_update_tau
        self.has_action_expert_branch = config.has_action_expert_branch
        self.has_value_expert_branch = config.has_value_expert_branch
        if self.td_learning and not self.has_value_expert_branch:
            raise ValueError("td_learning requires the value expert branch to be enabled.")

        paligemma_variant = config.action_obs_encoder.variant
        value_variant = config.value_obs_encoder.variant if config.value_obs_encoder else "gemma_300m"
        action_expert_variant = config.action_expert.variant if config.action_expert else "gemma_300m"
        paligemma_config = _gemma.get_config(paligemma_variant)
        value_config = _gemma.get_config(value_variant)
        action_expert_config = _gemma.get_config(action_expert_variant)
        fake_image = next(iter(config.fake_obs().images.values()))
        fake_action_image = _slice_image_history(fake_image, config.action_observation_horizon)
        fake_value_image = _slice_image_history(fake_image, config.value_observation_horizon)

        self.PaliGemma = None
        self.action_in_proj = None
        self.time_mlp_in = None
        self.time_mlp_out = None
        self.state_proj = None
        self.action_time_mlp_in = None
        self.action_time_mlp_out = None
        self.action_out_proj = None
        self.history_state_in_proj = None
        self.history_state_rnn = None
        self.history_value_in_proj = None
        self.history_value_rnn = None
        if self.has_action_expert_branch:
            # TODO: rewrite gemma in NNX. For now, use bridge.
            llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=[paligemma_config, action_expert_config],
                    embed_dtype=config.dtype,
                    adarms=config.pi05,
                )
            )
            llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
            action_encoder_type = config.action_obs_encoder.image_encoder.encoder_type
            action_temporal_interval = config.action_obs_encoder.image_encoder.temporal_attention_interval
            img = nnx_bridge.ToNNX(
                _make_image_encoder(
                    num_classes=paligemma_config.width,
                    dtype_mm=config.dtype,
                    encoder_type=action_encoder_type,
                    temporal_attention_interval=action_temporal_interval,
                )
            )
            img.lazy_init(
                _siglip_init_image(fake_action_image, encoder_type=action_encoder_type),
                train=False,
                rngs=rngs,
            )
            self.PaliGemma = nnx.Dict(llm=llm, img=img)
            self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            if config.pi05:
                self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
                self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            else:
                self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
                self.action_time_mlp_in = nnx.Linear(
                    2 * action_expert_config.width, action_expert_config.width, rngs=rngs
                )
                self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            if config.use_history_state_tokens:
                self.history_state_in_proj = nnx.Linear(config.action_dim, paligemma_config.width, rngs=rngs)
                self.history_state_rnn = HistoryGRU(paligemma_config.width, rngs=rngs)
            if config.use_history_value_tokens:
                self.history_value_in_proj = nnx.Linear(1, paligemma_config.width, rngs=rngs)
                self.history_value_rnn = HistoryGRU(paligemma_config.width, rngs=rngs)
            self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Separate PaliGemma model for value prediction
        if value_config.vocab_size == _gemma.PALIGEMMA_VOCAB_SIZE:
            value_embed_config = dataclasses.replace(value_config, embed_dim=paligemma_config.width)
        else:
            value_embed_config = value_config

        self.ValuePaliGemma = None
        self.value_text_proj = None
        self.value_params = None
        self.value_out_proj = None

        if self.has_value_expert_branch:
            # Value function: single-expert LLM
            value_llm = nnx_bridge.ToNNX(
                _gemma.Module(configs=[value_embed_config], embed_dtype=config.dtype, adarms=False)
            )
            value_llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])

            value_encoder_type = config.value_obs_encoder.image_encoder.encoder_type
            value_temporal_interval = config.value_obs_encoder.image_encoder.temporal_attention_interval
            value_img = nnx_bridge.ToNNX(
                _make_image_encoder(
                    num_classes=value_config.width,
                    dtype_mm=config.dtype,
                    encoder_type=value_encoder_type,
                    temporal_attention_interval=value_temporal_interval,
                )
            )
            value_img.lazy_init(
                _siglip_init_image(fake_value_image, encoder_type=value_encoder_type),
                train=False,
                rngs=rngs,
            )
            self.ValuePaliGemma = nnx.Dict(llm=value_llm, img=value_img)

            # Project embedder output (2048) to LLM hidden dim (1024) for encode.
            # This is always created when using PaliGemma 2B embedder with gemma_300m.
            if value_embed_config.embed_dim is not None and value_embed_config.embed_dim != value_config.width:
                self.value_text_proj = nnx.Linear(value_embed_config.embed_dim, value_config.width, rngs=rngs)
                logger.info(
                    "Value prompt projection enabled: %d -> %d",
                    value_embed_config.embed_dim,
                    value_config.width,
                )

            # Value function: learnable parameters for value head input
            self.value_params = nnx.Param(jnp.zeros((1, value_config.width)))
            self.value_out_proj = nnx.Linear(value_config.width, config.n_bins, rngs=rngs)

        # Target value function: single-expert LLM (TD learning mode only)
        self.TargetValuePaliGemma = None
        self.target_value_text_proj = None
        self.target_value_params = None
        self.target_value_out_proj = None
        self.target_update_step = None
        if self.td_learning:
            target_value_llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=[value_embed_config],
                    embed_dtype=config.dtype,
                    adarms=False,
                )
            )
            target_value_llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])
            target_value_img = nnx_bridge.ToNNX(
                _make_image_encoder(
                    num_classes=value_config.width,
                    dtype_mm=config.dtype,
                    encoder_type=config.value_image_encoder_type,
                    temporal_attention_interval=config.value_obs_encoder.image_encoder.temporal_attention_interval,
                )
            )
            target_value_img.lazy_init(
                _siglip_init_image(fake_value_image, encoder_type=config.value_image_encoder_type),
                train=False,
                rngs=rngs,
            )
            self.TargetValuePaliGemma = nnx.Dict(llm=target_value_llm, img=target_value_img)

            # Target update step counter - must be initialized BEFORE calling _polyak_update
            if value_embed_config.embed_dim is not None and value_embed_config.embed_dim != value_config.width:
                self.target_value_text_proj = nnx.Linear(value_embed_config.embed_dim, value_config.width, rngs=rngs)

            # Target value function: learnable parameters
            self.target_value_params = nnx.Param(jnp.zeros((1, value_config.width)))
            self.target_value_out_proj = nnx.Linear(value_config.width, config.n_bins, rngs=rngs)

            # Initialize target_update_step BEFORE calling _polyak_update
            self.target_update_step = nnx.Variable(jnp.array(0, dtype=jnp.int32))

            # Initialize target network to match online network at creation time
            # This avoids needing to do it during training (which causes JAX tracer issues)
            self._polyak_update_target_value_network(tau=1.0)
            # Set step to 1 after initialization (indicating initialization is complete)
            self.target_update_step.value = jnp.array(1, dtype=jnp.int32)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _require_action_branch(self, context: str) -> None:
        if not self.has_action_expert_branch:
            raise ValueError(f"{context} requires the action expert branch, but value_expert_branches excludes it.")

    def _require_value_branch(self, context: str) -> None:
        if not self.has_value_expert_branch:
            raise ValueError(f"{context} requires the value expert branch, but value_expert_branches excludes it.")

    def _require_target_value_branch(self, context: str) -> None:
        if not self.td_learning or self.TargetValuePaliGemma is None:
            raise ValueError(f"{context} requires TD-learning target value modules, but td_learning is disabled.")

    def _value_llm_embed(self, llm, tokens):
        return llm(tokens, method="embed")

    def _value_llm_forward(
        self,
        llm,
        tokens,
        *,
        mask,
        positions,
        kv_cache=None,
        decode: bool = False,
    ):
        return llm([tokens], mask=mask, positions=positions, kv_cache=kv_cache, decode=decode)

    @at.typecheck
    def embed_prefix(
        self,
        obs: _model.Observation,
        *,
        train: bool = False,
        rng: at.KeyArrayLike | None = None,
        conditional: bool = True,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        self._require_action_branch("embed_prefix")
        obs = _slice_observation_history(obs, self.action_observation_horizon)
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = _embed_image_tokens(
                self.PaliGemma.img,
                obs.images[name],
                encoder_type=self.action_image_encoder_type,
                image_mask=obs.image_masks[name],
                train=False,
            )

            tokens.append(image_tokens)
            input_mask.append(
                _image_token_mask(obs.image_masks[name], image_tokens, encoder_type=self.action_image_encoder_type)
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        if self.use_history_state_tokens:
            cprint("Policy is using history state tokens as input", "cyan")
            state_tokens = history_gru_token(
                self.history_state_in_proj, self.history_state_rnn, obs.state, obs.state_mask
            )
            tokens.append(state_tokens)
            input_mask.append(current_history_mask(mask=obs.state_mask, values=obs.state)[:, None])
            ar_mask += [False]

        if self.use_history_value_tokens:
            cprint("Policy is using history value predictions as input", "magenta")
            value_tokens = history_gru_token(
                self.history_value_in_proj,
                self.history_value_rnn,
                obs.value_prediction,
                obs.value_prediction_mask,
            )
            tokens.append(value_tokens)
            # Because the current value prediction is masked, but we still want the previous values.
            input_mask.append(
                current_history_mask(jnp.any(obs.value_prediction_mask, axis=-1), obs.value_prediction)[:, None]
            )
            ar_mask += [False]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]

        # Add advantage tokens if pi06 is enabled
        if self.pi06 and obs.tokenized_advantage is not None:
            if train:
                assert rng is not None, "Require rng for dropout in advantage conditioning"
            use_advantage = random.uniform(0, 1) < self.advantage_threshold if train else conditional
            if use_advantage:
                cprint("Advantage tokens are being included in the model input", "yellow")
                tokenized_advantage = self.PaliGemma.llm(obs.tokenized_advantage, method="embed")
                tokens.append(tokenized_advantage)
                input_mask.append(obs.tokenized_advantage_mask)
                # advantage tokens attend to all previous tokens
                ar_mask += [True] + ([False] * (tokenized_advantage.shape[1] - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _cfg_observation(self, obs: _model.Observation) -> _model.Observation:
        """Stack conditional and unconditional copies for one-pass CFG inference."""

        def repeat_batch(x):
            if x is None:
                return None
            return jnp.concatenate([x, x], axis=0)

        def repeat_dict(values):
            return {key: repeat_batch(value) for key, value in values.items()}

        tokenized_advantage_mask = None
        if obs.tokenized_advantage_mask is not None:
            tokenized_advantage_mask = jnp.concatenate(
                [obs.tokenized_advantage_mask, jnp.zeros_like(obs.tokenized_advantage_mask)],
                axis=0,
            )

        with at.disable_typechecking():
            return _model.Observation(
                images=repeat_dict(obs.images),
                image_masks=repeat_dict(obs.image_masks),
                state=repeat_batch(obs.state),
                state_mask=repeat_batch(obs.state_mask),
                value_prediction=repeat_batch(obs.value_prediction),
                value_prediction_mask=repeat_batch(obs.value_prediction_mask),
                tokenized_prompt=repeat_batch(obs.tokenized_prompt),
                tokenized_prompt_mask=repeat_batch(obs.tokenized_prompt_mask),
                tokenized_value_prompt=repeat_batch(obs.tokenized_value_prompt),
                tokenized_value_prompt_mask=repeat_batch(obs.tokenized_value_prompt_mask),
                tokenized_advantage=repeat_batch(obs.tokenized_advantage),
                tokenized_advantage_mask=tokenized_advantage_mask,
                token_ar_mask=repeat_batch(obs.token_ar_mask),
                token_loss_mask=repeat_batch(obs.token_loss_mask),
            )

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        self._require_action_branch("embed_suffix")
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_tokens, state_mask, state_ar_mask = _state_tokens(self.state_proj, obs)
            tokens.append(state_tokens)
            input_mask.append(state_mask)
            # image/language inputs do not attend to state or actions
            ar_mask += state_ar_mask

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @at.typecheck
    def embed_value_prefix(
        self,
        obs: _model.Observation,
        stop_img_gradient: bool = False,  # if True, stop the gradient of the image tokens
        precomputed_image_tokens=None,  # type: ignore[assignment] # dict[str, Float[Array, "b s emb"]] | None, skipped to avoid jaxtyping tracer error during JIT
    ) -> tuple[
        at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"] | at.Bool[at.Array, "b s"]
    ]:
        """Embed value function prefix tokens.

        This function is used for value function prediction.
        Input: Image + value_prompt (or prompt).

        Args:
            obs: The observation containing images and prompts.
            precomputed_image_tokens: Optional precomputed image tokens. If provided, skips
                SigLIP image encoding (saves expensive forward pass). Dict maps image key
                to (B, num_tokens, embed_dim) tensor.

        Returns:
            tokens: concatenated tokens [images, prompt]
            input_mask: attention input mask
            ar_mask: autoregressive mask
        """
        self._require_value_branch("embed_value_prefix")
        obs = _slice_observation_history(obs, self.value_observation_horizon)
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            if precomputed_image_tokens is not None and name in precomputed_image_tokens:
                image_tokens = precomputed_image_tokens[name]
            else:
                image_tokens, _ = _embed_image_tokens(
                    self.ValuePaliGemma.img,
                    obs.images[name],
                    encoder_type=self.value_image_encoder_type,
                    image_mask=obs.image_masks[name],
                    train=False,
                )

            if stop_img_gradient:
                image_tokens = jax.lax.stop_gradient(image_tokens)
            tokens.append(image_tokens)
            input_mask.append(
                _image_token_mask(obs.image_masks[name], image_tokens, encoder_type=self.value_image_encoder_type)
            )
            # image tokens attend to each other
            if obs.token_ar_mask is None:
                ar_mask += [False] * image_tokens.shape[1]
            else:
                ar_mask.append(0 * input_mask[-1])

        # add language (aka tokenized inputs)
        if obs.tokenized_value_prompt is not None:  # Removed proprio state from value input
            tokenized_inputs = self._value_llm_embed(self.ValuePaliGemma.llm, obs.tokenized_value_prompt)
            if self.value_text_proj is not None:
                tokenized_inputs = self.value_text_proj(tokenized_inputs)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_value_prompt_mask)
            if obs.token_ar_mask is not None:
                ar_mask.append(obs.token_ar_mask)
            else:
                ar_mask += [False] * tokenized_inputs.shape[1]
        elif obs.tokenized_prompt is not None:
            tokenized_inputs = self._value_llm_embed(self.ValuePaliGemma.llm, obs.tokenized_prompt)
            if self.value_text_proj is not None:
                tokenized_inputs = self.value_text_proj(tokenized_inputs)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            if obs.token_ar_mask is not None:
                ar_mask.append(obs.token_ar_mask)
            else:
                ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = (
            jnp.concatenate(ar_mask, axis=1).astype(jnp.bool_) if obs.token_ar_mask is not None else jnp.array(ar_mask)
        )
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_value_suffix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        self._require_value_branch("embed_value_suffix")
        tokens = self.value_params[:, None, :].repeat(obs.state.shape[0], axis=0)
        input_mask = jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_)
        ar_mask = jnp.array([True])
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_target_value_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        self._require_target_value_branch("embed_target_value_prefix")
        obs = _slice_observation_history(obs, self.value_observation_horizon)
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = _embed_image_tokens(
                self.TargetValuePaliGemma.img,
                obs.images[name],
                encoder_type=self.value_image_encoder_type,
                image_mask=obs.image_masks[name],
                train=False,
            )

            tokens.append(image_tokens)
            input_mask.append(
                _image_token_mask(obs.image_masks[name], image_tokens, encoder_type=self.value_image_encoder_type)
            )
            # image tokens attend to each other
            if obs.token_ar_mask is None:
                ar_mask += [False] * image_tokens.shape[1]
            else:
                ar_mask.append(0 * input_mask[-1])

        # add language (aka tokenized inputs)
        if obs.tokenized_value_prompt is not None:  # Removed proprio state from value input
            tokenized_inputs = self._value_llm_embed(self.TargetValuePaliGemma.llm, obs.tokenized_value_prompt)
            if self.target_value_text_proj is not None:
                tokenized_inputs = self.target_value_text_proj(tokenized_inputs)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_value_prompt_mask)
            if obs.token_ar_mask is not None:
                ar_mask.append(obs.token_ar_mask)
            else:
                ar_mask += [False] * tokenized_inputs.shape[1]
        elif obs.tokenized_prompt is not None:  # no specific proprio mask for target value expert
            tokenized_inputs = self._value_llm_embed(self.TargetValuePaliGemma.llm, obs.tokenized_prompt)
            if self.target_value_text_proj is not None:
                tokenized_inputs = self.target_value_text_proj(tokenized_inputs)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            if obs.token_ar_mask is not None:
                ar_mask.append(obs.token_ar_mask)
            else:
                ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = (
            jnp.concatenate(ar_mask, axis=1).astype(jnp.bool_) if obs.token_ar_mask is not None else jnp.array(ar_mask)
        )
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_target_value_suffix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        self._require_target_value_branch("embed_target_value_suffix")
        tokens = self.target_value_params[:, None, :].repeat(obs.state.shape[0], axis=0)
        input_mask = jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_)
        ar_mask = jnp.array([True])
        return tokens, input_mask, ar_mask

    def _polyak_update_module(self, target_module: nnx.Module, source_module: nnx.Module, tau: float) -> None:
        target_state = nnx.state(target_module)
        source_state = nnx.state(source_module)
        target_params = target_state.to_pure_dict()
        source_params = source_state.to_pure_dict()
        blended = jax.tree.map(lambda t, s: (1.0 - tau) * t + tau * s, target_params, source_params)
        # Use nnx.update to apply the blended parameters back to the target module
        # Note: nnx.state() returns a copy, so we must use nnx.update() to modify the original
        nnx.update(target_module, blended)

    def _polyak_update_target_value_network(self, tau: float) -> None:
        self._require_target_value_branch("_polyak_update_target_value_network")
        self._polyak_update_module(self.TargetValuePaliGemma, self.ValuePaliGemma, tau)
        if self.value_text_proj is not None and self.target_value_text_proj is not None:
            self._polyak_update_module(self.target_value_text_proj, self.value_text_proj, tau)
        self._polyak_update_module(self.target_value_out_proj, self.value_out_proj, tau)
        # Value function: update learnable value params
        self.target_value_params.value = (1.0 - tau) * self.target_value_params.value + tau * self.value_params.value

    def _maybe_update_target_value_network(self, *, train: bool) -> None:
        """Update target value network using Polyak averaging.

        This function is called from train.py post-optimizer to perform periodic EMA updates.
        The initialization (step 1) is handled separately via _initialize_target_value_network.
        """
        if not train:
            return
        self._require_target_value_branch("_maybe_update_target_value_network")
        self.target_update_step.value = self.target_update_step.value + 1
        step = self.target_update_step.value

        # Compute tau: tau=0 means no update, tau>0 means partial update
        # step == 1: initialization (should not happen here, handled by _initialize_target_value_network)
        # step % interval == 0: periodic update with target_update_tau
        # otherwise: no update (tau=0)
        tau = jnp.where(
            jnp.logical_and(step > 1, step % self.target_update_interval == 0),
            self.target_update_tau,
            jnp.array(0.0, dtype=jnp.float32),
        )

        # Apply Polyak update with computed tau (tau=0 means no change)
        self._polyak_update_target_value_network(tau=tau)

    def _value_support(self, num_atoms: int, dtype: jnp.dtype) -> at.Array:
        return _value_distribution.value_support(self.vmin, self.vmax, num_atoms, dtype)

    def _discrete_value_expectation(self, value_probs: at.Array) -> at.Array:
        return _value_distribution.discrete_value_expectation(value_probs, self.vmin, self.vmax)

    def _discounted_reward_sum(self, reward: _model.Reward, dtype: jnp.dtype) -> at.Array:
        discounts = self.gamma ** jnp.arange(self.action_horizon, dtype=dtype)
        return jnp.sum(reward * discounts[None, :], axis=1, keepdims=True)

    def _hl_gauss_value_distribution(self, target_value: at.Array, num_atoms: int, dtype: jnp.dtype) -> at.Array:
        return _value_distribution.hl_gauss_value_distribution(
            target_value,
            vmin=self.vmin,
            vmax=self.vmax,
            num_atoms=num_atoms,
            gaussian_sigma=self.gaussian_sigma,
            dtype=dtype,
        )

    def _discrete_td_value_loss(
        self,
        value_log_probs: at.Array,
        target_value_probs: at.Array,
        reward: _model.Reward,
        terminate: _model.Terminate | None = None,
    ) -> at.Array:
        td_target_dists = _value_distribution.distributional_td_target_distribution(
            target_value_probs,
            self._discounted_reward_sum(reward, target_value_probs.dtype),
            self.gamma**self.action_horizon,
            terminate,
            vmin=self.vmin,
            vmax=self.vmax,
            gaussian_sigma=self.gaussian_sigma,
            dtype=target_value_probs.dtype,
        )
        return -jnp.sum(
            jax.lax.stop_gradient(td_target_dists) * value_log_probs,
            axis=-1,
            keepdims=True,
        )

    def _bc_action_loss(
        self, observation, actions, success, noise_rng, time_rng, advantage_rng, filter_action_loss_by_success
    ):
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation, train=True, rng=advantage_rng)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if filter_action_loss_by_success and success is not None:
            action_loss = action_loss * success
        return action_loss

    def _mc_value_prediction(self, observation):
        value_prefix_tokens, value_prefix_mask, value_prefix_ar_mask = self.embed_value_prefix(observation)
        value_suffix_tokens, value_suffix_mask, value_suffix_ar_mask = self.embed_value_suffix(observation)
        value_input_mask = jnp.concatenate([value_prefix_mask, value_suffix_mask], axis=1)
        if value_prefix_ar_mask.ndim != value_suffix_ar_mask.ndim:
            value_suffix_ar_mask = value_suffix_ar_mask[None, :].repeat(value_prefix_ar_mask.shape[0], axis=0)
            value_ar_mask = jnp.concatenate([value_prefix_ar_mask, value_suffix_ar_mask], axis=1)
        else:
            value_ar_mask = jnp.concatenate([value_prefix_ar_mask, value_suffix_ar_mask], axis=0)
        value_attn_mask = make_attn_mask(value_input_mask, value_ar_mask)
        value_positions = jnp.cumsum(value_input_mask, axis=1) - 1
        value_tokens = jnp.concatenate([value_prefix_tokens, value_suffix_tokens], axis=1)
        value_output, _ = self._value_llm_forward(
            self.ValuePaliGemma.llm, value_tokens, mask=value_attn_mask, positions=value_positions
        )
        value_suffix_out = jnp.split(value_output[0], [value_prefix_tokens.shape[1]], axis=1)[1]
        value_pred = self.value_out_proj(value_suffix_out)
        value_pred = jax.nn.log_softmax(value_pred.astype(jnp.float32), axis=-1)
        return value_pred

    def _mc_value_loss(self, observation, value, data_source):
        value_pred = self._mc_value_prediction(observation)
        value_labels = value[:, None, :].astype(jnp.float32)
        safe_value_log_probs = jnp.where(value_labels > 0, value_pred, 0.0)
        mc_value_loss = jnp.sum(-value_labels * safe_value_log_probs, axis=-1)  # (bsz, 1)
        if data_source is not None:
            mc_value_loss = mc_value_loss * (
                1.0 - data_source
            )  # (bsz, 1), only offline expert data receives mc value loss
        return mc_value_loss

    def _td_value_loss(self, observation, next_observation, reward, terminate):
        # Target network update is now handled in train.py to ensure proper ordering:
        # 1. Pre-gradient: initialize target = online (tau=1.0) on first step
        # 2. Post-optimizer: update target = online with EMA (tau=target_update_tau)
        # This ensures target is updated from the latest online params after gradient step.
        td_value_prefix_tokens, td_value_prefix_mask, td_value_prefix_ar_mask = self.embed_value_prefix(
            observation, stop_img_gradient=False
        )
        td_value_suffix_tokens, td_value_suffix_mask, td_value_suffix_ar_mask = self.embed_value_suffix(observation)
        td_value_input_mask = jnp.concatenate([td_value_prefix_mask, td_value_suffix_mask], axis=1)
        if td_value_prefix_ar_mask.ndim != td_value_suffix_ar_mask.ndim:
            td_value_suffix_ar_mask = td_value_suffix_ar_mask[None, :].repeat(td_value_prefix_ar_mask.shape[0], axis=0)
            td_value_ar_mask = jnp.concatenate([td_value_prefix_ar_mask, td_value_suffix_ar_mask], axis=1)
        else:
            td_value_ar_mask = jnp.concatenate([td_value_prefix_ar_mask, td_value_suffix_ar_mask], axis=0)
        td_value_attn_mask = make_attn_mask(td_value_input_mask, td_value_ar_mask)
        td_value_positions = jnp.cumsum(td_value_input_mask, axis=1) - 1
        td_value_tokens = jnp.concatenate([td_value_prefix_tokens, td_value_suffix_tokens], axis=1)
        td_value_output, _ = self._value_llm_forward(
            self.ValuePaliGemma.llm, td_value_tokens, mask=td_value_attn_mask, positions=td_value_positions
        )
        td_value_suffix_out = jnp.split(td_value_output[0], [td_value_prefix_tokens.shape[1]], axis=1)[1]
        td_value_pred = self.value_out_proj(td_value_suffix_out)
        td_value_pred = jax.nn.log_softmax(td_value_pred.astype(jnp.float32), axis=-1)

        target_value_prefix_tokens, target_value_prefix_mask, target_value_prefix_ar_mask = (
            self.embed_target_value_prefix(next_observation)
        )
        target_value_suffix_tokens, target_value_suffix_mask, target_value_suffix_ar_mask = (
            self.embed_target_value_suffix(next_observation)
        )
        target_value_input_mask = jnp.concatenate([target_value_prefix_mask, target_value_suffix_mask], axis=1)
        if target_value_prefix_ar_mask.ndim != target_value_suffix_ar_mask.ndim:
            target_value_suffix_ar_mask = target_value_suffix_ar_mask[None, :].repeat(
                target_value_prefix_ar_mask.shape[0], axis=0
            )
            target_value_ar_mask = jnp.concatenate([target_value_prefix_ar_mask, target_value_suffix_ar_mask], axis=1)
        else:
            target_value_ar_mask = jnp.concatenate([target_value_prefix_ar_mask, target_value_suffix_ar_mask], axis=0)
        target_value_attn_mask = make_attn_mask(target_value_input_mask, target_value_ar_mask)
        target_value_positions = jnp.cumsum(target_value_input_mask, axis=1) - 1
        target_value_tokens = jnp.concatenate([target_value_prefix_tokens, target_value_suffix_tokens], axis=1)
        target_value_output, _ = self._value_llm_forward(
            self.TargetValuePaliGemma.llm,
            target_value_tokens,
            mask=target_value_attn_mask,
            positions=target_value_positions,
        )
        target_value_suffix_out = jnp.split(target_value_output[0], [target_value_prefix_tokens.shape[1]], axis=1)[1]
        target_value_pred = self.target_value_out_proj(target_value_suffix_out)

        # Extract target value for bootstrapping: (bsz, 1, 1) or (bsz, 1, n_bins)
        target_value_probs = nnx.softmax(target_value_pred, axis=-1)  # (bsz, 1, n_bins)

        # Compute multi-step TD target
        # reward shape: (bsz, action_horizon)
        td_value_loss = self._discrete_td_value_loss(
            td_value_pred[:, 0, :],
            target_value_probs[:, 0, :],
            reward,
            terminate,
        )  # (bsz, 1)
        return td_value_loss

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        value: _model.Values | None = None,
        next_obs: _model.Observation | None = None,
        success: _model.Success | None = None,
        reward: _model.Reward | None = None,
        next_actions: _model.Actions | None = None,
        control_flag: _model.ControlFlag | None = None,
        data_source: _model.DataSource | None = None,
        terminate: _model.Terminate | None = None,
        *,
        train: bool = False,
        train_type: Set[_model.TrainingType] = {_model.TrainingType.BC},  # default to only BC training
        filter_action_loss_by_success: bool = True,
    ) -> tuple[Union[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b 1"]], ...]:
        preprocess_rng, noise_rng, time_rng, advantage_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        next_observation = None
        if next_obs is not None:
            next_observation = _model.preprocess_observation(preprocess_rng, next_obs, train=train)

        assert len(train_type) > 0, "No training type specified"

        requires_action_branch = _model.TrainingType.BC in train_type
        requires_value_branch = any(
            t in train_type
            for t in (
                _model.TrainingType.TD_LEARNING,
                _model.TrainingType.VALUE_MC_LEARNING,
            )
        )

        if requires_action_branch:
            self._require_action_branch("compute_loss")
        if requires_value_branch:
            self._require_value_branch("compute_loss")

        if _model.TrainingType.TD_LEARNING in train_type and next_observation is None:
            raise ValueError("td_learning requires next_obs.")

        loss = 0.0
        all_losses = []
        if _model.TrainingType.BC in train_type:
            action_loss = self._bc_action_loss(
                observation, actions, success, noise_rng, time_rng, advantage_rng, filter_action_loss_by_success
            )
            loss += action_loss
            all_losses.append(action_loss)
        if _model.TrainingType.TD_LEARNING in train_type:
            assert next_observation is not None
            td_value_loss = self._td_value_loss(observation, next_observation, reward, terminate)
            loss += td_value_loss
            all_losses.append(td_value_loss)
        if _model.TrainingType.VALUE_MC_LEARNING in train_type:
            mc_value_loss = self._mc_value_loss(observation, value, data_source)
            loss += mc_value_loss
            all_losses.append(mc_value_loss)

        return loss, *all_losses

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        cfg_guidance_weight: float | at.Float[at.Array, ""] = 1.0,
        enable_rtc: bool = False,
        prefix_actions: jax.Array | None = None,
        inference_delay: int | None = None,
        prefix_attention_horizon: int | None = None,
        max_guidance_weight: float | None = None,
        prefix_attention_schedule: PrefixAttentionSchedule = "exp",
    ) -> _model.Actions:
        """Sample actions, optionally guiding them with a normalized previous chunk.

        RTC accepts an unbatched [action_horizon, action_dim] prefix for a single
        observation. Smaller action dimensions are padded to the model width.
        JIT callers must make enable_rtc and prefix_attention_schedule static;
        cfg_guidance_weight is also static for ordinary sampling.
        """
        self._require_action_branch("sample_actions")
        if not isinstance(enable_rtc, bool):
            raise ValueError("enable_rtc must be a static Python bool.")
        observation = _model.preprocess_observation(None, observation, train=False)
        # Diffusion time runs from noise at t=1 to the action distribution at t=0.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if enable_rtc:
            if batch_size != 1:
                raise ValueError("RTC requires batch_size=1.")
            if prefix_actions is None:
                raise ValueError("prefix_actions is required when enable_rtc=True.")
            if inference_delay is None or prefix_attention_horizon is None or max_guidance_weight is None:
                raise ValueError("RTC requires inference_delay, prefix_attention_horizon and max_guidance_weight.")
            prefix_actions = jnp.asarray(prefix_actions)
            if (
                prefix_actions.ndim != 2
                or prefix_actions.shape[0] != self.action_horizon
                or not 0 < prefix_actions.shape[1] <= self.action_dim
            ):
                raise ValueError(
                    f"RTC prefix_actions must have shape ({self.action_horizon}, 1..{self.action_dim}), "
                    f"got {prefix_actions.shape}."
                )
            prefix_actions = jnp.concatenate(
                [
                    prefix_actions[None, ...],
                    jnp.zeros((batch_size, self.action_horizon, self.action_dim - prefix_actions.shape[-1])),
                ],
                axis=2,
            )
            prefix_weights = get_prefix_weights(
                inference_delay,
                prefix_attention_horizon + inference_delay,
                self.action_horizon,
                prefix_attention_schedule,
            )

        if not enable_rtc and not isinstance(cfg_guidance_weight, int | float):
            raise ValueError("cfg_guidance_weight must be a static Python float for sample_actions CFG.")
        use_cfg = (
            self.pi06
            and observation.tokenized_advantage is not None
            and (not isinstance(cfg_guidance_weight, int | float) or cfg_guidance_weight != 1.0)
        )
        model_observation = self._cfg_observation(observation) if use_cfg else observation
        model_batch_size = batch_size * 2 if use_cfg else batch_size

        # Encode the observation once; both sampling modes reuse its KV cache.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(model_observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def velocity(x_t, time):
            model_x_t = jnp.concatenate([x_t, x_t], axis=0) if use_cfg else x_t
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                model_observation, model_x_t, jnp.broadcast_to(time, model_batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                model_batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            if use_cfg:
                conditional_v_t, unconditional_v_t = jnp.split(v_t, 2, axis=0)
                v_t = cfg_guidance_weight * conditional_v_t - (cfg_guidance_weight - 1.0) * unconditional_v_t

            return v_t

        def step(carry):
            x_t, time = carry
            if enable_rtc:
                v_t = rtc_velocity(velocity, x_t, time)
                v_t = guide_velocity(x_t, v_t, time, prefix_actions, prefix_weights, max_guidance_weight)
            else:
                v_t = velocity(x_t, time)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            # Robust to floating-point error.
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @override
    def predict_value(
        self,
        observation: _model.Observation,
        actions: _model.Actions | None = None,
        *,
        precomputed_image_tokens=None,  # type: ignore[assignment] # skipped to avoid jaxtyping tracer error during JIT
    ) -> tuple[_model.Values, jnp.ndarray] | jnp.ndarray:
        """Predict a categorical value distribution using the value expert."""
        self._require_value_branch("predict_value")
        observation = _model.preprocess_observation(None, observation, train=False)

        # Value function forward pass with single-expert LLM

        value_prefix_tokens, value_prefix_mask, value_prefix_ar_mask = self.embed_value_prefix(
            observation, precomputed_image_tokens=precomputed_image_tokens
        )
        value_suffix_tokens, value_suffix_mask, value_suffix_ar_mask = self.embed_value_suffix(observation)
        if value_prefix_ar_mask.ndim != value_suffix_ar_mask.ndim:
            value_suffix_ar_mask = value_suffix_ar_mask[None, :].repeat(value_prefix_ar_mask.shape[0], axis=0)
            value_ar_mask = jnp.concatenate([value_prefix_ar_mask, value_suffix_ar_mask], axis=1)
        else:
            value_ar_mask = jnp.concatenate([value_prefix_ar_mask, value_suffix_ar_mask], axis=0)
        value_input_mask = jnp.concatenate([value_prefix_mask, value_suffix_mask], axis=1)
        value_attn_mask = make_attn_mask(value_input_mask, value_ar_mask)
        value_positions = jnp.cumsum(value_input_mask, axis=1) - 1
        value_tokens = jnp.concatenate([value_prefix_tokens, value_suffix_tokens], axis=1)
        value_output, _ = self._value_llm_forward(
            self.ValuePaliGemma.llm, value_tokens, mask=value_attn_mask, positions=value_positions
        )
        value_suffix_out = jnp.split(value_output[0], [value_prefix_tokens.shape[1]], axis=1)[1]
        value_pred = self.value_out_proj(value_suffix_out)
        value_pred = nnx.softmax(value_pred, axis=-1)
        return value_pred[:, -1, :]
