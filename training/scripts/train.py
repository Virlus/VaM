"""Initialize with offline dataset: action-expert behavior cloning and value-expert progress initialization."""

import dataclasses
import functools
import logging
import platform
import shutil
import threading
from typing import Any, Set

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import vam.models.model as _model
import vam.shared.array_typing as at
from vam.shared.logging_utils import init_cli_logging
import vam.shared.nnx_utils as nnx_utils
from vam.training.augmentation_visualizer import maybe_visualize_augmentations
import vam.training.checkpoints as _checkpoints
import vam.training.config as _config
import vam.training.data_loader as _data_loader
import vam.training.optimizer as _optimizer
import vam.training.sharding as sharding
import vam.training.utils as training_utils
import vam.training.validation as _validation


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = training_utils.cast_frozen_params_to_bfloat16(params, config.freeze_filter)

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = training_utils.load_and_validate_partial_params(
        config.weight_loader, train_state_shape.params.to_pure_dict()
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    train_type: Set[_model.TrainingType],
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[
        _model.Observation,
        _model.Actions,
        _model.Values | None,
        _model.Observation | None,
        _model.Success | None,
        _model.Reward | None,
        _model.Actions | None,
        _model.ControlFlag | None,
        _model.DataSource | None,
        _model.Terminate | None,
    ],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
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
        train_type: Set[_model.TrainingType] = {_model.TrainingType.BC},
    ):
        losses = model.compute_loss(
            rng,
            observation=observation,
            actions=actions,
            value=value,
            next_obs=next_obs,
            success=success,
            reward=reward,
            next_actions=next_actions,
            control_flag=control_flag,
            data_source=data_source,
            terminate=terminate,
            train=True,
            train_type=train_type,
            filter_action_loss_by_success=config.filter_action_loss_by_success,
        )
        assert len(losses) == len(train_type) + 1, "Number of losses and training types mismatch"
        total_loss, *atomic_losses = losses
        return jnp.mean(total_loss), atomic_losses

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, value, next_obs, success, reward, next_actions, control_flag, data_source, terminate = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_losses), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model,
        train_rng,
        observation,
        actions,
        value,
        next_obs,
        success,
        reward,
        next_actions,
        control_flag,
        data_source,
        terminate,
        train_type,
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)

    # Apply target network Polyak update AFTER optimizer step to ensure the target
    # is updated from the latest online network params (not overwritten by optimizer).
    # This is only relevant for TD learning which uses target value/Q networks.
    if _model.TrainingType.TD_LEARNING in train_type:
        # VAM uses _maybe_update_target_value_network
        # Pi0QFunction uses _maybe_update_target_q_network
        if hasattr(model, "_maybe_update_target_value_network"):
            model._maybe_update_target_value_network(train=True)
        elif hasattr(model, "_maybe_update_target_q_network"):
            model._maybe_update_target_q_network(train=True)

    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: (
                    state.ema_decay * old + (1 - state.ema_decay) * new
                    if jnp.issubdtype(new.dtype, jnp.floating)
                    else new
                ),
                state.ema_params,
                new_params,
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    sorted_train_types = [curr_type for curr_type in _model.TrainingType if curr_type in train_type]
    for curr_train_type, curr_loss in zip(sorted_train_types, aux_losses):
        info[curr_train_type.value + "_loss"] = jnp.mean(curr_loss)
    return new_state, info


def copy_checkpoint_async(source_dir, target_dir, step):
    """Asynchronously copy checkpoints"""

    def _copy():
        try:
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy checkpoint directory
            logging.info(f"[Backstage] Start copying checkpoint {step} from {source_dir} to {target_dir}")
            shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
            logging.info(f"[Backstage] Successfully copied checkpoint {step} to {target_dir}")
        except Exception as e:
            logging.error(f"[Backstage] Copy checkpoint {step} failed: {e}")

    thread = threading.Thread(target=_copy, daemon=True)
    thread.start()
    return thread


def main(config: _config.TrainConfig):
    init_cli_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    training_utils.init_wandb_run(
        config,
        resuming=resuming,
        enabled=config.wandb_enabled,
        log_code_path=epath.Path(__file__).parent.parent,
        clear_proxy_env=True,
    )

    # Track copying threads at the backstage
    copy_threads = []
    validation_episode_refs: list[_validation.ValidationEpisodeRef] | None = None
    validation_loaders: list[_validation.ValidationLoader] | None = None
    validation_video_threads: list[threading.Thread] = []

    if config.validation.enabled:
        validation_episode_refs = _validation.resolve_validation_episode_refs(config)
        validation_loaders = _validation.prepare_validation_loaders(config, validation_episode_refs)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        skip_norm_stats=not config.use_norm_stats,
    )
    data_iter = training_utils.threaded_prefetch(
        iter(data_loader),
        prefetch_size=config.prefetch_size,
        transform=(lambda item: jax.tree.map(jax.device_put, item)) if config.device_prefetch else None,
        error_message="Training prefetch worker failed",
    )
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    observation, actions, _, _, _, _, _, _, _, _ = batch
    if config.startup_previews:

        def _preview_frames(image):
            image = np.array(image)
            if image.ndim == 4:
                return [image[t] for t in range(image.shape[0])]
            return [image]

        images_to_log = []
        for sample_idx in range(min(5, len(next(iter(observation.images.values()))))):
            camera_histories = [_preview_frames(img[sample_idx]) for img in observation.images.values()]
            history_len = max(len(history) for history in camera_histories)
            for history_idx in range(history_len):
                camera_frames = [history[min(history_idx, len(history) - 1)] for history in camera_histories]
                caption = (
                    f"sample={sample_idx}, history={history_idx - history_len + 1}"
                    if history_len > 1
                    else f"sample={sample_idx}"
                )
                images_to_log.append(wandb.Image(np.concatenate(camera_frames, axis=1), caption=caption))
        wandb.log({"camera_views": images_to_log}, step=0)

        # Visualize augmentations at step 0 (before training starts)
        maybe_visualize_augmentations(
            step=0,
            observation=observation,
            checkpoint_dir=str(config.checkpoint_dir),
            save_interval=getattr(config, "aug_vis_interval", 1000),
            num_augmentations=getattr(config, "num_augmentations", 20),
        )

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    full_checkpoint_template = training_utils.build_full_value_expert_checkpoint_template(config)

    train_type = set(_model.TrainingType(x) for x in config.train_type)

    ptrain_step = jax.jit(
        functools.partial(train_step, config, train_type),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            checkpoint_params_override = training_utils.build_checkpoint_params_override(
                train_state, full_checkpoint_template
            )
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                step,
                params_override=checkpoint_params_override,
            )
            checkpoint_manager.wait_until_finished()
            _validation.run_in_process_validation(
                config,
                step=step,
                train_state=train_state,
                validation_loaders=validation_loaders,
                mesh=mesh,
                data_sharding=data_sharding,
                active_video_threads=validation_video_threads,
            )

    logging.info("Waiting for checkpoint manager to finish...")
    checkpoint_manager.wait_until_finished()
    if validation_video_threads:
        _validation.join_validation_video_threads(validation_video_threads)

    # Waiting for all the copying threads to finish
    if copy_threads:
        logging.info(f"Waiting for {len(copy_threads)} backstage copying threads...")
        for i, thread in enumerate(copy_threads):
            thread.join()
            logging.info(f"Copying thread {i + 1}/{len(copy_threads)} finished")
        logging.info("All copying threads have been finished")


if __name__ == "__main__":
    main(_config.cli())
