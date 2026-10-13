# Utilities for discrete value-distribution targets.

import jax
import jax.numpy as jnp

from vam.shared import array_typing as at


def value_support(vmin: float, vmax: float, num_atoms: int, dtype: jnp.dtype) -> at.Array:
    """Return the fixed support used by a discrete value head."""
    if num_atoms < 2:
        raise ValueError(f"num_atoms must be >= 2, got {num_atoms}")
    if vmin >= vmax:
        raise ValueError(f"vmin must be less than vmax, got vmin={vmin}, vmax={vmax}")
    return jnp.linspace(vmin, vmax, num_atoms, dtype=dtype)


def discrete_value_expectation(value_probs: at.Array, vmin: float, vmax: float) -> at.Array:
    """Compute E[V] from probabilities over the fixed support."""
    support = value_support(vmin, vmax, value_probs.shape[-1], value_probs.dtype)
    return jnp.sum(support * value_probs, axis=-1)


def hl_gauss_value_distribution(
    target_value: at.Array,
    *,
    vmin: float,
    vmax: float,
    num_atoms: int,
    gaussian_sigma: float,
    dtype: jnp.dtype,
) -> at.Array:
    """Project scalar targets to a local Gaussian distribution on the fixed support."""
    support = value_support(vmin, vmax, num_atoms, dtype)
    diffs = target_value - support[None, :]
    logits = -0.5 * jnp.square(diffs / gaussian_sigma)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    unnormalized = jnp.exp(logits)
    weights = unnormalized.astype(dtype)
    row_sums = jnp.clip(weights.sum(axis=-1, keepdims=True), 1e-8, None)
    return weights / row_sums


def project_distribution_to_support(
    probabilities: at.Array,
    target_support_values: at.Array,
    *,
    vmin: float,
    vmax: float,
    dtype: jnp.dtype,
) -> at.Array:
    """Project a categorical distribution at arbitrary locations back to the fixed support."""
    num_atoms = probabilities.shape[-1]
    if num_atoms < 2:
        raise ValueError(f"num_atoms must be >= 2, got {num_atoms}")
    if vmin >= vmax:
        raise ValueError(f"vmin must be less than vmax, got vmin={vmin}, vmax={vmax}")
    support_delta = jnp.asarray((vmax - vmin) / (num_atoms - 1), dtype=dtype)
    clipped_values = jnp.clip(target_support_values, vmin, vmax)
    bin_positions = jnp.clip((clipped_values - vmin) / support_delta, 0.0, num_atoms - 1)
    lower_idx = jnp.floor(bin_positions).astype(jnp.int32)
    upper_idx = jnp.ceil(bin_positions).astype(jnp.int32)

    same_bin = lower_idx == upper_idx
    lower_weight = jnp.where(same_bin, 1.0, upper_idx.astype(dtype) - bin_positions)
    upper_weight = jnp.where(same_bin, 0.0, bin_positions - lower_idx.astype(dtype))

    probabilities = probabilities.astype(dtype)
    lower_mass = probabilities * lower_weight
    upper_mass = probabilities * upper_weight
    projected = jnp.sum(
        lower_mass[..., None] * jax.nn.one_hot(lower_idx, num_atoms, dtype=dtype)
        + upper_mass[..., None] * jax.nn.one_hot(upper_idx, num_atoms, dtype=dtype),
        axis=-2,
    )
    row_sums = jnp.clip(projected.sum(axis=-1, keepdims=True), 1e-8, None)
    return projected / row_sums


def distributional_td_target_distribution(
    target_value_probs: at.Array,
    reward_offset: at.Array,
    discount: float,
    terminate: at.Array | None,
    *,
    vmin: float,
    vmax: float,
    gaussian_sigma: float,
    dtype: jnp.dtype,
) -> at.Array:
    """Build a TD target that preserves the target-network categorical distribution when not terminal."""
    num_atoms = target_value_probs.shape[-1]
    support = value_support(vmin, vmax, num_atoms, dtype)
    shifted_support = reward_offset + discount * support[None, :]
    bootstrapped_distribution = project_distribution_to_support(
        target_value_probs,
        shifted_support,
        vmin=vmin,
        vmax=vmax,
        dtype=dtype,
    )
    if terminate is None:
        return bootstrapped_distribution

    terminal_distribution = hl_gauss_value_distribution(
        reward_offset,
        vmin=vmin,
        vmax=vmax,
        num_atoms=num_atoms,
        gaussian_sigma=gaussian_sigma,
        dtype=dtype,
    )
    terminal_mask = jnp.asarray(terminate, dtype=dtype)
    if terminal_mask.ndim == target_value_probs.ndim - 1:
        terminal_mask = terminal_mask[..., None]
    return (1.0 - terminal_mask) * bootstrapped_distribution + terminal_mask * terminal_distribution
