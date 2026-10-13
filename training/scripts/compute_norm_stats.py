"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import vam.models.model as _model
import vam.shared.normalize as normalize
import vam.training.config as _config
import vam.training.data_loader as _data_loader
import vam.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_lowdim_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if not data_config.effective_repo_ids():
        raise ValueError("Data config must have a repo_id or repo_ids")
    dataset = _data_loader.create_lowdim_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *_data_loader.build_lowdim_norm_transforms(data_config, model_config),
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None, action_horizon: int | None = None):
    """Compute norm stats for ``config_name``.

    ``action_horizon`` overrides the horizon used to build the low-dim dataset. Leave it as ``None``
    to use ``config.model.action_horizon``; pass ``1`` for the async / value-learning workflow that
    computes single-frame stats. Next-observation keys (``next_state``/``next_actions``) are included
    automatically when the data config sets ``use_next_obs``.
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    horizon = config.model.action_horizon if action_horizon is None else action_horizon

    data_loader, num_batches = create_lowdim_torch_dataloader(
        data_config, horizon, config.batch_size, config.model, config.num_workers, max_frames
    )

    keys = ["state", "actions"]
    if data_config.use_next_obs:
        keys.extend(["next_state", "next_actions"])
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    if data_config.asset_id is None:
        raise ValueError("Data config must define asset_id to write norm stats")
    output_path = config.assets_dirs / data_config.asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
