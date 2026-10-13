from __future__ import annotations

import dataclasses
import functools
import gc
import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import vam.models.model as _model
from vam.shared import array_typing as at
from vam.training.async_control._common import _read_json
from vam.training.async_control._common import _resolve_path
import vam.training.config as _config
import vam.training.utils as training_utils
import vam.training.weight_loaders as weight_loaders


def _restore_params_from_params_dir(params_dir: str | Path, template_params: at.Params) -> at.Params:
    loaded_params = _model.restore_params(params_dir, restore_type=np.ndarray)
    if hasattr(template_params, "to_pure_dict"):
        restored_pure = weight_loaders.merge_partial_params_into_reference(
            loaded_params,
            template_params.to_pure_dict(),
        )
        restored_params = jax.tree.map(lambda value: value, template_params)
        restored_params.replace_by_pure_dict(restored_pure)
        return restored_params
    return weight_loaders.merge_partial_params_into_reference(loaded_params, template_params)


def validate_checkpoint_ref(checkpoint_ref: dict[str, Any]) -> Path:
    checkpoint_path = checkpoint_ref.get("checkpoint_path") or checkpoint_ref.get("params_path")
    if checkpoint_path is None:
        raise ValueError(f"Checkpoint switch ref is missing checkpoint_path/params_path: {checkpoint_ref}")
    checkpoint_path = _resolve_path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent if checkpoint_path.name == "params" else checkpoint_path
    params_dir = checkpoint_path if checkpoint_path.name == "params" else checkpoint_path / "params"
    if not params_dir.exists():
        raise FileNotFoundError(f"Checkpoint params path does not exist: {params_dir}")
    done_path = checkpoint_dir / "DONE"
    if checkpoint_ref.get("require_done", True) and checkpoint_path.name != "params" and not done_path.exists():
        raise FileNotFoundError(f"Checkpoint is not marked DONE: {done_path}")
    manifest_path = checkpoint_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        expected_id = checkpoint_ref.get("checkpoint_id")
        if expected_id is not None and manifest.get("checkpoint_id") not in (None, expected_id):
            raise ValueError(
                f"Checkpoint manifest id mismatch: registry={expected_id!r}, manifest={manifest.get('checkpoint_id')!r}"
            )
    return params_dir


def _put_like_reference(params: at.Params, reference_params: at.Params) -> at.Params:
    def _put_leaf(value, reference):
        sharding = getattr(reference, "sharding", None)
        if sharding is None:
            return value
        return jax.device_put(value, sharding)

    return jax.tree.map(_put_leaf, params, reference_params)


def _copy_jax_array_buffers(params: at.Params) -> at.Params:
    """Copy JAX array leaves so params and EMA params do not alias donated buffers."""

    def _copy_leaf(value):
        if hasattr(value, "dtype"):
            return jnp.array(value, copy=True)
        return value

    return jax.tree.map(_copy_leaf, params)


def finalize_checkpoint_switch(
    state: training_utils.TrainState,
    *,
    context: str,
    reclaim_memory: bool = False,
    clear_jax_caches: bool = False,
) -> training_utils.TrainState:
    """Synchronize a checkpoint switch and optionally release transient memory.

    Checkpoint rebasing creates a short-lived overlap between the old train state, restored params, and the newly
    reset optimizer/EMA state. Python garbage collection is safe after the switched state is ready, but
    ``jax.clear_caches()`` also evicts the compiled training step. Keep JAX cache eviction opt-in so a checkpoint
    rebase with unchanged shapes and train type can reuse its executable on the next step.
    """
    jax.block_until_ready(state)
    if reclaim_memory:
        gc.collect()
        if clear_jax_caches:
            logging.info("Reclaimed transient Python checkpoint-switch memory after %s.", context)
        else:
            logging.info("Reclaimed transient Python checkpoint-switch memory after %s; retained JAX caches.", context)
    if clear_jax_caches:
        jax.clear_caches()
        logging.warning("Cleared JAX caches after %s; the next training step may recompile.", context)
    return state


def _count_like_step(step: Any, reference: Any) -> Any:
    if not hasattr(reference, "dtype"):
        return step
    count = jnp.asarray(step, dtype=reference.dtype)
    return jnp.broadcast_to(count, np.shape(reference))


def _set_schedule_counts_to_step(opt_state: Any, step: Any) -> Any:
    """Reset optimizer moments while keeping only LR-schedule counters aligned with the global training step."""

    if hasattr(opt_state, "_fields"):
        is_schedule_state = type(opt_state).__name__ == "ScaleByScheduleState"
        values = {
            name: (
                _count_like_step(step, getattr(opt_state, name))
                if name == "count" and is_schedule_state
                else _set_schedule_counts_to_step(getattr(opt_state, name), step)
            )
            for name in opt_state._fields
        }
        return type(opt_state)(**values)
    if dataclasses.is_dataclass(opt_state) and not isinstance(opt_state, type):
        is_schedule_state = type(opt_state).__name__ == "ScaleByScheduleState"
        values = {
            field.name: (
                _count_like_step(step, getattr(opt_state, field.name))
                if field.name == "count" and is_schedule_state
                else _set_schedule_counts_to_step(getattr(opt_state, field.name), step)
            )
            for field in dataclasses.fields(opt_state)
        }
        return dataclasses.replace(opt_state, **values)
    if isinstance(opt_state, tuple):
        return type(opt_state)(_set_schedule_counts_to_step(value, step) for value in opt_state)
    if isinstance(opt_state, list):
        return [_set_schedule_counts_to_step(value, step) for value in opt_state]
    if isinstance(opt_state, dict):
        return {key: _set_schedule_counts_to_step(value, step) for key, value in opt_state.items()}
    return opt_state


def _reset_optimizer_state_at_step(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    params: at.Params,
    step: Any,
) -> Any:
    opt_state = state.tx.init(params.filter(config.trainable_filter))
    return _set_schedule_counts_to_step(opt_state, step)


def _switch_train_state_params_impl(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    restored_params: at.Params,
    *,
    reset_optimizer: bool,
    reset_ema: bool,
    preserve_global_step: bool,
) -> training_utils.TrainState:
    new_step = state.step if preserve_global_step else jnp.zeros_like(state.step)
    opt_state = (
        _reset_optimizer_state_at_step(config, state, restored_params, new_step) if reset_optimizer else state.opt_state
    )
    if state.ema_decay is None:
        ema_params = None
    elif reset_ema:
        ema_params = _copy_jax_array_buffers(restored_params)
    else:
        logging.info(
            "Preserving local EMA params across checkpoint switch; only live params are replaced by the checkpoint."
        )
        ema_params = state.ema_params
    return dataclasses.replace(
        state,
        step=new_step,
        params=restored_params,
        opt_state=opt_state,
        ema_params=ema_params,
    )


def _switch_train_state_params(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    restored_params: at.Params,
    *,
    reset_optimizer: bool,
    reset_ema: bool,
    preserve_global_step: bool,
    state_sharding: jax.sharding.Sharding | None,
) -> training_utils.TrainState:
    if state_sharding is None:
        return _switch_train_state_params_impl(
            config,
            state,
            restored_params,
            reset_optimizer=reset_optimizer,
            reset_ema=reset_ema,
            preserve_global_step=preserve_global_step,
        )

    params_sharding = getattr(state_sharding, "params", None)

    @functools.partial(
        jax.jit,
        donate_argnums=(0, 1),
        in_shardings=(state_sharding, params_sharding),
        out_shardings=state_sharding,
    )
    def _jit_switch(
        current_state: training_utils.TrainState,
        new_params: at.Params,
    ) -> training_utils.TrainState:
        return _switch_train_state_params_impl(
            config,
            current_state,
            new_params,
            reset_optimizer=reset_optimizer,
            reset_ema=reset_ema,
            preserve_global_step=preserve_global_step,
        )

    return _jit_switch(state, restored_params)


def _switch_train_state_to_current_ema_impl(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    *,
    reset_optimizer: bool,
    reset_ema: bool,
    preserve_global_step: bool,
) -> training_utils.TrainState:
    restored_params = state.ema_params if state.ema_params is not None else state.params
    return _switch_train_state_params_impl(
        config,
        state,
        restored_params,
        reset_optimizer=reset_optimizer,
        reset_ema=reset_ema,
        preserve_global_step=preserve_global_step,
    )


def switch_train_state_to_current_ema(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    *,
    checkpoint_id: str | None,
    reset_optimizer: bool,
    reset_ema: bool,
    preserve_global_step: bool,
    state_sharding: jax.sharding.Sharding | None = None,
) -> training_utils.TrainState:
    logging.info(
        "Rebasing train state to current EMA checkpoint %s "
        "(reset_optimizer=%s, reset_ema=%s, preserve_global_step=%s, optimizer_count_step=%s)",
        checkpoint_id,
        reset_optimizer,
        reset_ema,
        preserve_global_step,
        int(state.step),
    )
    if state_sharding is None:
        return _switch_train_state_to_current_ema_impl(
            config,
            state,
            reset_optimizer=reset_optimizer,
            reset_ema=reset_ema,
            preserve_global_step=preserve_global_step,
        )

    @functools.partial(jax.jit, donate_argnums=(0,), in_shardings=(state_sharding,), out_shardings=state_sharding)
    def _jit_switch(current_state: training_utils.TrainState) -> training_utils.TrainState:
        return _switch_train_state_to_current_ema_impl(
            config,
            current_state,
            reset_optimizer=reset_optimizer,
            reset_ema=reset_ema,
            preserve_global_step=preserve_global_step,
        )

    return _jit_switch(state)


def switch_train_state_checkpoint(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    checkpoint_ref: dict[str, Any],
    *,
    reset_optimizer: bool,
    reset_ema: bool,
    preserve_global_step: bool,
    state_sharding: jax.sharding.Sharding | None = None,
) -> training_utils.TrainState:
    params_dir = validate_checkpoint_ref(checkpoint_ref)

    restored_params = _restore_params_from_params_dir(params_dir, state.params)
    restored_params = training_utils.cast_frozen_params_to_bfloat16(restored_params, config.freeze_filter)
    restored_params = _put_like_reference(restored_params, state.params)
    new_step = state.step if preserve_global_step else 0
    logging.info(
        "Switching train state to checkpoint %s "
        "(reset_optimizer=%s, reset_ema=%s, preserve_global_step=%s, optimizer_count_step=%s)",
        checkpoint_ref.get("checkpoint_id", params_dir),
        reset_optimizer,
        reset_ema,
        preserve_global_step,
        int(new_step),
    )
    return _switch_train_state_params(
        config,
        state,
        restored_params,
        reset_optimizer=reset_optimizer,
        reset_ema=reset_ema,
        preserve_global_step=preserve_global_step,
        state_sharding=state_sharding,
    )
