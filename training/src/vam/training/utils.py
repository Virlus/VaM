from collections.abc import Callable, Iterator
import dataclasses
import logging
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, TypeVar

from etils import epath
from flax import nnx
from flax import struct
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from vam.models import model as _model
from vam.models import vam_config as _vam_config
from vam.shared import array_typing as at
from vam.shared import nnx_utils
from vam.training import config as _config
from vam.training import weight_loaders as _weight_loaders

T = TypeVar("T")
U = TypeVar("U")


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


def init_wandb_run(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    enabled: bool = True,
    log_code_path: str | Path | None = None,
    clear_proxy_env: bool = False,
) -> None:
    if not enabled:
        wandb.init(mode="disabled")
        return

    if clear_proxy_env:
        for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
            os.environ.pop(var, None)

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    run_id_path = ckpt_dir / "wandb_id.txt"
    merge_config = config.async_training.checkpoint_publisher.merge
    wandb_name = config.exp_name
    wandb_group = None
    wandb_job_type = None
    wandb_tags = None
    if merge_config is not None:
        wandb_group = config.exp_name
        wandb_job_type = merge_config.role
        wandb_name = f"{config.exp_name}/{merge_config.role}"
        wandb_tags = ["async", "recap", "split", merge_config.role]
    if resuming:
        run_id = run_id_path.read_text().strip()
        wandb.init(
            id=run_id,
            resume="must",
            project=config.project_name,
            group=wandb_group,
            job_type=wandb_job_type,
            tags=wandb_tags,
        )
    else:
        wandb.init(
            name=wandb_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            group=wandb_group,
            job_type=wandb_job_type,
            tags=wandb_tags,
        )
        run_id_path.write_text(wandb.run.id)

    if log_code_path is not None:
        wandb.run.log_code(epath.Path(log_code_path))


def log_wandb_startup(config: _config.TrainConfig) -> None:
    if wandb.run is None:
        return
    merge_config = config.async_training.checkpoint_publisher.merge
    role = merge_config.role if merge_config is not None else "default"
    payload = {
        "control/run_started": 1,
        f"control/{role}/run_started": 1,
        f"control/{role}/configured_num_train_steps": config.num_train_steps,
        f"control/{role}/configured_log_interval": config.log_interval,
        f"control/{role}/configured_batch_size": config.batch_size,
    }
    try:
        wandb.log(payload, step=0)
        wandb.run.summary["control/role"] = role
        wandb.run.summary["control/config_name"] = config.name
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not log W&B startup metadata: %s", exc)


def resolve_checkpoint_params_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "params" if (path / "params").exists() else path


def _has_shape_dtype_struct(x: Any) -> bool:
    return any(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree_util.tree_leaves(x))


def filter_unloaded_shape_dtype_structs(params: at.Params) -> at.Params:
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(params).items() if not _has_shape_dtype_struct(v)}
    )


def load_and_validate_partial_params(loader: _weight_loaders.WeightLoader, reference_params: Any) -> at.Params:
    loaded_params = loader.load(reference_params)
    try:
        at.check_pytree_equality(
            expected=reference_params,
            got=loaded_params,
            check_shapes=True,
            check_dtypes=True,
        )
    except ValueError as exc:
        logging.warning("Weights validation warning: %s", exc)
        logging.warning("Proceeding with partial loading.")
    return filter_unloaded_shape_dtype_structs(loaded_params)


def cast_frozen_params_to_bfloat16(params: nnx.State, freeze_filter: Any) -> nnx.State:
    return nnx_utils.state_map(
        params,
        freeze_filter,
        lambda p: nnx_utils.cast_variable_state_dtype(p, jnp.bfloat16),
    )


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")


def build_full_value_expert_checkpoint_template(config: _config.TrainConfig) -> at.Params | None:
    """For slim VAMConfig training, build a full-branch param template on CPU.

    Returns None when not applicable (non-value-expert config or already full branches).
    The returned tree lives on host memory so it can be safely merged with sharded training
    params during checkpoint saves without triggering device-placement conflicts.
    """
    if not isinstance(config.model, _vam_config.VAMConfig):
        return None
    # Full-branch training normally needs no rehydration. Split async training is
    # different: frozen parameters (for example history-value encoders) may be
    # omitted from the role checkpoint even when both branches exist in the model.
    # The merged checkpoint is consumed by inference, so it must always be full.
    split_merge_enabled = config.async_training.checkpoint_publisher.merge is not None
    if config.model.has_action_expert_branch and config.model.has_value_expert_branch and not split_merge_enabled:
        return None

    # Build a full-branch config (both action and value) for the checkpoint template.
    full_model_config = dataclasses.replace(
        config.model,
        action_expert=config.model.action_expert or _vam_config.ActionExpertConfig(pi05=config.model.pi05),
        value_expert=config.model.value_expert or _vam_config.ValueExpertConfig(),
    )
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("Expected at least one CPU device for checkpoint template construction.")

    with jax.default_device(cpu_devices[0]):
        full_model = full_model_config.create(jax.random.key(0))
        full_reference_params = nnx.state(full_model).to_pure_dict()
    full_template = load_and_validate_partial_params(config.weight_loader, full_reference_params)
    # Keep template params on host memory to avoid device-placement conflicts when
    # mixing with sharded training params during checkpoint rehydration.
    full_template = jax.tree.map(np.asarray, jax.device_get(full_template))
    logging.info(
        "Enabled full-checkpoint rehydration for slim value expert training (action_branch=%s, value_branch=%s).",
        config.model.has_action_expert_branch,
        config.model.has_value_expert_branch,
    )
    return full_template


def build_checkpoint_params_override(
    state: TrainState,
    full_template: at.Params | None,
) -> at.Params | None:
    """Merge current (EMA or live) params into the full-branch template for checkpoint saving.

    Returns None when no template is provided (i.e. full-branch or non-value-expert training).
    """
    if full_template is None:
        return None
    params_for_inference = state.ema_params if state.ema_params is not None else state.params
    # Orbax async save can fail if arrays from different shard placements are mixed
    # in one pytree. Materialize value-only params on host before merge.
    partial_params_host = jax.tree.map(np.asarray, jax.device_get(params_for_inference.to_pure_dict()))
    merged = _weight_loaders.merge_partial_params_into_reference(partial_params_host, full_template)
    return jax.tree.map(np.asarray, merged)


def threaded_prefetch(
    iterator: Iterator[T],
    *,
    prefetch_size: int,
    transform: Callable[[T], U] | None = None,
    error_message: str = "Background prefetch worker failed",
    debug_name: str | None = None,
    generation: Callable[[], int] | None = None,
) -> Iterator[T | U]:
    debug_enabled = os.environ.get("VAM_ASYNC_DATALOADER_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    debug_every = max(1, int(os.environ.get("VAM_ASYNC_DATALOADER_DEBUG_EVERY", "10")))
    debug_name = debug_name or "prefetch"

    if prefetch_size <= 0:
        try:
            if transform is None:
                yield from iterator
                return
            for index, item in enumerate(iterator, start=1):
                transform_t0 = time.monotonic()
                transformed = transform(item)
                transform_ms = (time.monotonic() - transform_t0) * 1e3
                if debug_enabled and index % debug_every == 0:
                    logging.info(
                        "DATALOADER_PREFETCH name=%s mode=sync index=%d transform_ms=%.0f",
                        debug_name,
                        index,
                        transform_ms,
                    )
                yield transformed
            return
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    sentinel = object()
    error_sentinel = object()
    item_queue: queue.Queue[Any] = queue.Queue(maxsize=prefetch_size)
    worker_errors: list[BaseException] = []
    stop_event = threading.Event()

    def _put(item: Any) -> tuple[bool, float]:
        put_t0 = time.monotonic()
        while not stop_event.is_set():
            try:
                item_queue.put(item, timeout=0.1)
                return True, time.monotonic() - put_t0
            except queue.Full:
                continue
        return False, time.monotonic() - put_t0

    def _worker() -> None:
        try:
            index = 0
            while not stop_event.is_set():
                version = generation() if generation is not None else None
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                index += 1
                if stop_event.is_set():
                    break
                transform_t0 = time.monotonic()
                transformed = transform(item) if transform is not None else item
                transform_elapsed = time.monotonic() - transform_t0
                ok, put_elapsed = _put((version, transformed))
                if debug_enabled and index % debug_every == 0:
                    logging.info(
                        "DATALOADER_PREFETCH name=%s mode=worker index=%d transform_ms=%.0f queue_put_ms=%.0f "
                        "queue_size=%d/%d",
                        debug_name,
                        index,
                        transform_elapsed * 1e3,
                        put_elapsed * 1e3,
                        item_queue.qsize(),
                        prefetch_size,
                    )
                if not ok:
                    break
            _put(sentinel)
        except Exception as exc:
            worker_errors.append(exc)
            logging.exception(error_message)
            _put(error_sentinel)
        finally:
            # Only the owning worker may close an iterator that it is currently advancing.
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    try:
        index = 0
        while True:
            get_t0 = time.monotonic()
            item = item_queue.get()
            get_elapsed = time.monotonic() - get_t0
            if item is sentinel:
                break
            if item is error_sentinel:
                raise RuntimeError(error_message) from worker_errors[0]
            index += 1
            if debug_enabled and index % debug_every == 0:
                logging.info(
                    "DATALOADER_PREFETCH name=%s mode=consumer index=%d queue_get_wait_ms=%.0f queue_size=%d/%d",
                    debug_name,
                    index,
                    get_elapsed * 1e3,
                    item_queue.qsize(),
                    prefetch_size,
                )
            version, value = item
            if generation is None or version == generation():
                yield value
    finally:
        stop_event.set()
        while True:
            try:
                item_queue.get_nowait()
            except queue.Empty:
                break
        thread.join(timeout=1.0)
