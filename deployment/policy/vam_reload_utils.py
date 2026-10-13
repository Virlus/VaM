"""Helpers for VAM checkpoint reloads."""

import gc
from pathlib import Path

from utils.logger_config import logger


def resolve_checkpoint_norm_stats(checkpoint_dir, asset_id):
    """Select stats only from this checkpoint, with an unambiguous fallback."""
    assets = Path(checkpoint_dir) / "assets"
    expected = assets / asset_id / "norm_stats.json" if asset_id else None
    if expected is not None and expected.is_file():
        logger.info(f"VAM normalization stats: {expected}")
        return expected

    candidates = sorted(p for p in assets.rglob("norm_stats.json") if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No norm_stats.json found in checkpoint assets: {assets}; configured asset_id={asset_id!r}"
        )
    if len(candidates) != 1:
        paths = ", ".join(str(p.relative_to(assets)) for p in candidates)
        raise ValueError(
            f"Ambiguous checkpoint normalization stats: configured asset_id={asset_id!r} "
            f"did not match; found {len(candidates)} files in {assets}: {paths}. "
            "Set the matching data.assets.asset_id explicitly."
        )
    selected = candidates[0]
    logger.warning(
        f"VAM configured norm stats path {expected} is unavailable; "
        f"using the checkpoint's only norm stats file: {selected}"
    )
    return selected


def clear_accelerator_caches() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.debug(f"Failed to clear torch CUDA cache during policy reload: {exc}")

    try:
        import jax

        if hasattr(jax, "clear_caches"):
            jax.clear_caches()
    except Exception as exc:
        logger.debug(f"Failed to clear JAX caches during policy reload: {exc}")
