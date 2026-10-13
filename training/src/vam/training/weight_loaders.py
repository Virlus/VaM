import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import vam.models.gemma as _gemma
import vam.models.model as _model
import vam.shared.array_typing as at
import vam.shared.download as download

logger = logging.getLogger("vam")


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class TDLearningWeightLoader(WeightLoader):
    """Loads a checkpoint and initializes target value weights from online value weights."""

    base_params_path: str

    def load(self, params: at.Params) -> at.Params:
        base_params = _model.restore_params(download.maybe_download(self.base_params_path), restore_type=np.ndarray)

        flat_base = flax.traverse_util.flatten_dict(base_params, sep="/")
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        target_value_params = {}
        for k, v in flat_base.items():
            new_key = _target_value_key(k)
            if new_key is not None and new_key in flat_ref:
                target_value_params[new_key] = v
                logger.info("Loading %s to %s", k, new_key)

        target_value_expert_params = flax.traverse_util.unflatten_dict(target_value_params, sep="/")
        params = _merge_params(base_params, params, missing_regex=".*")
        return _merge_params(target_value_expert_params, params, missing_regex=".*")


def _target_value_key(key: str) -> str | None:
    """Map an online value parameter key to its target-network key."""
    if key.startswith("Value"):
        return "TargetValue" + key[len("Value") :]
    if key.startswith("value_"):
        return "target_value_" + key[len("value_") :]
    return None


def _decoupled_td_value_keys(key: str) -> tuple[str, ...]:
    """Map pretrained value image encoder params into decoupled TD image encoders."""
    if not key.startswith("ValuePaliGemma/img/"):
        return ()
    suffix = key[len("ValuePaliGemma/img/") :]
    return (f"TDValuePaliGemma/img/{suffix}", f"TargetTDValuePaliGemma/img/{suffix}")


def _flatten_dict_by_str_preserving_paths(
    params: at.Params,
) -> tuple[dict[str, object], dict[str, tuple[object, ...]]]:
    flat = flax.traverse_util.flatten_dict(params)
    values = {}
    paths = {}
    for path, value in flat.items():
        str_path = "/".join(str(part) for part in path)
        if str_path in values:
            raise ValueError(f"Flattened parameter path collision for {str_path!r}.")
        values[str_path] = value
        paths[str_path] = path
    return values, paths


def _merge_params_by_str_path(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    flat_ref, ref_paths = _flatten_dict_by_str_preserving_paths(params)
    flat_loaded, _ = _flatten_dict_by_str_preserving_paths(loaded_params)

    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result_path = ref_paths[k]
            ref_value = flat_ref[k]
            if v is None or ref_value is None:
                result[result_path] = ref_value if v is None else v
                continue
            if getattr(v, "shape", None) != getattr(ref_value, "shape", None):
                logger.info(
                    "Skipping %s due to shape mismatch: loaded=%s reference=%s",
                    k,
                    getattr(v, "shape", None),
                    getattr(ref_value, "shape", None),
                )
                continue
            result[result_path] = v.astype(ref_value.dtype) if v.dtype != ref_value.dtype else v

    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        result_path = ref_paths[k]
        if result_path not in result:
            result[result_path] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref_value = flat_ref[k]
            if v is None or ref_value is None:
                result[k] = ref_value if v is None else v
                continue
            if getattr(v, "shape", None) != getattr(ref_value, "shape", None):
                logger.info(
                    "Skipping %s due to shape mismatch: loaded=%s reference=%s",
                    k,
                    getattr(v, "shape", None),
                    getattr(ref_value, "shape", None),
                )
                continue
            result[k] = v.astype(ref_value.dtype) if v.dtype != ref_value.dtype else v

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def merge_partial_params_into_reference(partial_params: at.Params, reference_params: at.Params) -> at.Params:
    """Merge partial params into a full reference tree.

    Keys present in ``partial_params`` override corresponding values in ``reference_params``.
    All missing keys are preserved from ``reference_params``.
    """
    flat_ref = flax.traverse_util.flatten_dict(reference_params, sep="/")
    flat_partial = flax.traverse_util.flatten_dict(partial_params, sep="/")
    merged = dict(flat_ref)
    merged.update({k: v for k, v in flat_partial.items() if k in flat_ref})
    return flax.traverse_util.unflatten_dict(merged, sep="/")


def _pi05_value_llm_init_strategy(
    *,
    paligemma_variant: _gemma.Variant,
    value_variant: _gemma.Variant,
    action_expert_variant: _gemma.Variant,
) -> str:
    if value_variant == paligemma_variant:
        return "paligemma"
    if value_variant == action_expert_variant:
        return "action_expert"
    raise ValueError(
        "Pi05 value expert loading requires value_variant to match either paligemma_variant or "
        f"action_expert_variant, got paligemma_variant={paligemma_variant}, "
        f"value_variant={value_variant}, action_expert_variant={action_expert_variant}."
    )


def _init_pi05_value_expert_flat_params(
    flat_loaded: dict[str, np.ndarray],
    *,
    params_path: str,
    paligemma_variant: _gemma.Variant,
    value_variant: _gemma.Variant,
    action_expert_variant: _gemma.Variant,
) -> dict[str, np.ndarray]:
    strategy = _pi05_value_llm_init_strategy(
        paligemma_variant=paligemma_variant,
        value_variant=value_variant,
        action_expert_variant=action_expert_variant,
    )
    logger.info(
        "Initializing Pi05 value expert from %s with %s strategy "
        "(paligemma_variant=%s, value_variant=%s, action_expert_variant=%s)",
        params_path,
        strategy,
        paligemma_variant,
        value_variant,
        action_expert_variant,
    )

    new_params = {}
    for k, v in flat_loaded.items():
        if k.startswith("PaliGemma/img/"):
            new_key = k.replace("PaliGemma/img/", "ValuePaliGemma/img/")
            if "head" in new_key:
                continue
            new_params[new_key] = v
        elif strategy == "paligemma" and k.startswith("PaliGemma/llm/"):
            new_key = k.replace("PaliGemma/llm/", "ValuePaliGemma/llm/")
            new_params[new_key] = v
        elif strategy == "action_expert":
            if k.startswith("PaliGemma/llm/embedder/"):
                new_key = k.replace("PaliGemma/llm/", "ValuePaliGemma/llm/")
                new_params[new_key] = v
            elif k.startswith("PaliGemma/llm/"):
                parts = k.split("/")
                if any(p.endswith("_1") and not p.isdigit() for p in parts):
                    new_parts = [p[:-2] if (p.endswith("_1") and not p.isdigit()) else p for p in parts]
                    new_key = "/".join(new_parts).replace("PaliGemma/llm/", "ValuePaliGemma/llm/")
                    new_params[new_key] = v

    return new_params


@dataclasses.dataclass(frozen=True)
class Pi05ValueExpertWeightLoader(WeightLoader):
    """Loads Pi05 checkpoints and initializes value-expert weights from config-selected variants."""

    params_path: str
    paligemma_variant: _gemma.Variant
    value_variant: _gemma.Variant
    action_expert_variant: _gemma.Variant

    def _load_direct_checkpoint(self, loaded_params: at.Params, params: at.Params) -> at.Params:
        logger.info("Loading Pi05 value-expert checkpoint directly from %s", self.params_path)
        return _merge_params(loaded_params, params, missing_regex=".*")

    def _llm_init_strategy(self) -> str:
        return _pi05_value_llm_init_strategy(
            paligemma_variant=self.paligemma_variant,
            value_variant=self.value_variant,
            action_expert_variant=self.action_expert_variant,
        )

    def _init_value_expert_flat_params(self, flat_loaded: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return _init_pi05_value_expert_flat_params(
            flat_loaded,
            params_path=self.params_path,
            paligemma_variant=self.paligemma_variant,
            value_variant=self.value_variant,
            action_expert_variant=self.action_expert_variant,
        )

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        if any(k.startswith("ValuePaliGemma/") for k in flat_loaded):
            return self._load_direct_checkpoint(loaded_params, params)

        new_params = self._init_value_expert_flat_params(flat_loaded)
        value_expert_params = flax.traverse_util.unflatten_dict(new_params, sep="/")
        params = _merge_params(loaded_params, params, missing_regex=".*")
        return _merge_params(value_expert_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class Pi0ValueExpertFromPi05BaseWeightLoader(Pi05ValueExpertWeightLoader):
    """Policy-specific pi05-base initializer for value-expert policies."""


@dataclasses.dataclass(frozen=True)
class Pi05ValueStateHistoryWeightLoader(WeightLoader):
    """Loads a pi05 value-expert model with optional siglip_video action encoder migration.

    All keys that exist in both the checkpoint and the reference params are loaded
    directly. For missing keys corresponding to siglip_video per-layer blocks
    (encoderblock_i / temporal_encoderblock_i), weights are sliced from the
    scanned single-frame siglip encoderblock in the checkpoint.

    If the checkpoint is a base pi05 checkpoint without ValuePaliGemma, the value
    expert is initialized with the same variant-aware mapping as
    Pi05ValueExpertWeightLoader. When the reference model has TargetValuePaliGemma,
    it is initialized from the final ValuePaliGemma weights. All other missing keys
    fall back to their reference values.
    """

    params_path: str
    paligemma_variant: _gemma.Variant = "gemma_2b"
    value_variant: _gemma.Variant = "gemma_300m"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_value_expert = {}
        if not any(k.startswith("ValuePaliGemma/") for k in flat_loaded):
            flat_value_expert = _init_pi05_value_expert_flat_params(
                flat_loaded,
                params_path=self.params_path,
                paligemma_variant=self.paligemma_variant,
                value_variant=self.value_variant,
                action_expert_variant=self.action_expert_variant,
            )
        flat_siglip_source = {**flat_loaded, **flat_value_expert}

        result = {}
        for k, v in flat_loaded.items():
            if k in flat_ref:
                ref_value = flat_ref[k]
                if v is None or ref_value is None:
                    result[k] = ref_value if v is None else v
                    continue
                if getattr(v, "shape", None) != getattr(ref_value, "shape", None):
                    logger.info(
                        "Skipping %s due to shape mismatch: loaded=%s reference=%s",
                        k,
                        getattr(v, "shape", None),
                        getattr(ref_value, "shape", None),
                    )
                    continue
                result[k] = v.astype(ref_value.dtype) if v.dtype != ref_value.dtype else v

        for k, v in flat_value_expert.items():
            if k in flat_ref and k not in result:
                ref_value = flat_ref[k]
                if v is None or ref_value is None:
                    result[k] = ref_value if v is None else v
                    continue
                if getattr(v, "shape", None) != getattr(ref_value, "shape", None):
                    logger.info(
                        "Skipping %s due to shape mismatch: loaded=%s reference=%s",
                        k,
                        getattr(v, "shape", None),
                        getattr(ref_value, "shape", None),
                    )
                    continue
                result[k] = v.astype(ref_value.dtype) if v.dtype != ref_value.dtype else v

        for k in flat_ref:
            if k not in result:
                video_value = self._init_siglip_video_param_from_scanned_siglip(k, flat_siglip_source)
                if video_value is not None:
                    ref_value = flat_ref[k]
                    result[k] = (
                        video_value.astype(ref_value.dtype) if video_value.dtype != ref_value.dtype else video_value
                    )
                else:
                    result[k] = flat_ref[k]

        self._copy_value_paligemma_to_existing_target(result, flat_ref)

        return flax.traverse_util.unflatten_dict(result, sep="/")

    @staticmethod
    def _copy_value_paligemma_to_existing_target(result: dict[str, object], flat_ref: dict[str, object]) -> None:
        """Initializes an optional target network from concrete online value weights."""
        prefix = "ValuePaliGemma/"
        target_prefix = "TargetValuePaliGemma/"
        for key, value in tuple(result.items()):
            if not key.startswith(prefix) or not isinstance(value, np.ndarray):
                continue
            target_key = target_prefix + key[len(prefix) :]
            if target_key not in flat_ref:
                continue
            target_ref = flat_ref[target_key]
            if getattr(value, "shape", None) != getattr(target_ref, "shape", None):
                logger.info(
                    "Skipping %s to %s due to shape mismatch: online=%s target=%s",
                    key,
                    target_key,
                    getattr(value, "shape", None),
                    getattr(target_ref, "shape", None),
                )
                continue
            target_dtype = getattr(target_ref, "dtype", None)
            result[target_key] = value.astype(target_dtype) if value.dtype != target_dtype else value
            logger.info("Loading %s to %s", key, target_key)

    @staticmethod
    def _init_siglip_video_param_from_scanned_siglip(key: str, flat_loaded: dict[str, np.ndarray]) -> np.ndarray | None:
        """Slices layer i from a scanned SigLIP encoderblock for siglip_video per-layer params."""
        match = re.fullmatch(r"(.*/img/Transformer/)(encoderblock|temporal_encoderblock)_(\d+)/(.*)", key)
        if match is None:
            return None
        prefix, _, layer, suffix = match.groups()
        source = flat_loaded.get(f"{prefix}encoderblock/{suffix}")
        if source is None:
            return None
        layer_index = int(layer)
        if source.ndim == 0 or source.shape[0] <= layer_index:
            return None
        return source[layer_index]


def _llm_key_is_expert1(key: str) -> bool:
    """Detect a base dual-expert LLM key belonging to expert 1 (action expert)."""
    return any(part.endswith("_1") and not part.isdigit() for part in key.split("/"))


def _llm_key_strip_expert1(key: str) -> str:
    """Strip the `_1` expert-1 marker so a key matches the canonical (expert 0) form."""
    return "/".join(part[:-2] if (part.endswith("_1") and not part.isdigit()) else part for part in key.split("/"))


def _init_pi05_q_function_flat_params(
    flat_base: dict[str, np.ndarray],
    *,
    params_path: str,
    paligemma_variant: _gemma.Variant,
    q_variant: _gemma.Variant,
    action_expert_variant: _gemma.Variant,
) -> dict[str, np.ndarray]:
    """Build QPaliGemma / TargetQPaliGemma params from a pi05 base checkpoint.

    The source expert for each Q LLM sub-network is selected from the config-supplied variants:
      - Q observation prefix expert (QPaliGemma expert 0): chosen by `q_variant`.
      - Q action-network expert (QPaliGemma expert 1): chosen by `action_expert_variant`.
    Each strategy maps to either the base 2B PaliGemma LLM (expert 0) or the base 300M action
    expert (expert 1). The SigLIP image encoder and the token embedder are always copied from the
    observation PaliGemma (2B). Shape-incompatible selections are skipped later by `_merge_params`.
    """
    q_strategy = _pi05_value_llm_init_strategy(
        paligemma_variant=paligemma_variant,
        value_variant=q_variant,
        action_expert_variant=action_expert_variant,
    )
    action_strategy = _pi05_value_llm_init_strategy(
        paligemma_variant=paligemma_variant,
        value_variant=action_expert_variant,
        action_expert_variant=action_expert_variant,
    )
    logger.info(
        "Initializing Pi05 Q-function from %s (paligemma_variant=%s, q_variant=%s, "
        "action_expert_variant=%s); q-prefix strategy=%s, q-action strategy=%s",
        params_path,
        paligemma_variant,
        q_variant,
        action_expert_variant,
        q_strategy,
        action_strategy,
    )

    # Canonical (expert-0 form) -> value for each base expert, plus the original expert-1 key.
    base_expert0: dict[str, np.ndarray] = {}
    base_expert1: dict[str, np.ndarray] = {}
    canon_to_expert1_rel: dict[str, str] = {}
    for k, v in flat_base.items():
        if not k.startswith("PaliGemma/llm/"):
            continue
        rel = k[len("PaliGemma/llm/") :]
        if rel.startswith("embedder/"):
            continue
        if _llm_key_is_expert1(k):
            canon = _llm_key_strip_expert1(rel)
            base_expert1[canon] = v
            canon_to_expert1_rel[canon] = rel
        else:
            base_expert0[rel] = v

    new_params: dict[str, np.ndarray] = {}
    q_prefixes = ("QPaliGemma/llm/", "TargetQPaliGemma/llm/")
    img_prefixes = ("QPaliGemma/img/", "TargetQPaliGemma/img/")

    for k, v in flat_base.items():
        # Image encoder: always from the observation PaliGemma SigLIP (skip the head projector).
        if k.startswith("PaliGemma/img/") and "head" not in k:
            for prefix in img_prefixes:
                new_params[k.replace("PaliGemma/img/", prefix)] = v
        # Token embedder: always from the observation PaliGemma (matches the 2B embed_dim).
        elif k.startswith("PaliGemma/llm/embedder/"):
            for prefix in q_prefixes:
                new_params[k.replace("PaliGemma/llm/", prefix)] = v

    # Q observation prefix expert (QPaliGemma expert 0, no `_1` marker).
    source_prefix = base_expert0 if q_strategy == "paligemma" else base_expert1
    for canon, v in source_prefix.items():
        for prefix in q_prefixes:
            new_params[prefix + canon] = v

    # Q action-network expert (QPaliGemma expert 1, keeps the `_1` marker).
    source_action = base_expert0 if action_strategy == "paligemma" else base_expert1
    for canon, v in source_action.items():
        expert1_rel = canon_to_expert1_rel.get(canon)
        if expert1_rel is None:
            continue
        for prefix in q_prefixes:
            new_params[prefix + expert1_rel] = v

    return new_params
