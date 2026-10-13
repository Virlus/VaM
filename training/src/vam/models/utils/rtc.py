"""Prefix weighting and the deployed RTC action-guidance rule."""

import jax
import jax.numpy as jnp

from vam.models.vam_config import PrefixAttentionSchedule


def rtc_velocity(velocity, x_t, time):
    """Preserve the deployed RTC denoiser's AD primal without evaluating a pullback.

    Removing linearization changes the BF16 forward lowering on H20, even though
    the old sampler did not use its VJP correction. Keep that numerical path;
    prefix guidance still uses the direct error below, not a Jacobian product.
    """

    def denoiser(sample):
        v_t = velocity(sample[None, ...], time)[0]
        return sample - v_t * time, v_t

    def primal_velocity(sample):
        _, _, v_t = jax.vjp(denoiser, sample, has_aux=True)
        return v_t

    return jax.vmap(primal_velocity)(x_t)


def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule) -> jax.Array:
    """Hold the delayed prefix, taper its remaining horizon, then ignore the tail.

    ``end`` is exclusive and takes precedence when it is smaller than ``start``.
    For linear start=2, end=6, total=10 the weights are
    [1, 1, 4/5, 3/5, 2/5, 1/5, 0, 0, 0, 0].
    """
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule in {"linear", "exp"}:
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)


def guide_velocity(x_t, v_t, time, prefix_actions, prefix_weights, max_guidance_weight):
    """Apply the existing direct prefix-error correction, including gripper channels.

    The previous sampler did not use its VJP correction or gripper mask in its
    return value. The denoiser's AD primal is preserved separately by rtc_velocity.
    """
    x_0 = x_t - v_t * time
    error = (prefix_actions - x_0) * prefix_weights[:, None]
    inv_r2 = (time**2 + (1 - time) ** 2) / (time**2)
    c = jnp.nan_to_num(time / (1 - time), posinf=max_guidance_weight)
    guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
    return v_t - guidance_weight * error
