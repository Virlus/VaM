"""Async streaming control plane.

Split into three cohesive submodules (plus shared JSON/path utils):

- ``dataset_view``        — dataset-view schema parsing, application, and the file watcher
- ``checkpoint_publish``  — checkpoint publishing, value/action param merging, publisher/switch watchers
- ``train_state_switch``  — train-state param surgery for checkpoint / EMA switches

This package preserves the original flat ``async_control`` import surface: every
symbol previously accessed as ``async_control.X`` (including the ``_``-prefixed
helpers consumed by ``scripts/``) is re-exported here.
"""

from vam.training.async_control._common import _atomic_write_json
from vam.training.async_control._common import _read_json
from vam.training.async_control._common import _resolve_path
from vam.training.async_control.checkpoint_publish import _VALUE_PARAM_PREFIXES
from vam.training.async_control.checkpoint_publish import CheckpointPublisher
from vam.training.async_control.checkpoint_publish import CheckpointSwitchWatcher
from vam.training.async_control.checkpoint_publish import _checkpoint_dir_from_ref
from vam.training.async_control.checkpoint_publish import _copy_merged_assets
from vam.training.async_control.checkpoint_publish import _copytree_data_only
from vam.training.async_control.checkpoint_publish import _flatten_dict
from vam.training.async_control.checkpoint_publish import _matches_prefix
from vam.training.async_control.checkpoint_publish import _merge_value_params_into_action_params
from vam.training.async_control.checkpoint_publish import _merged_checkpoint_ref_is_complete
from vam.training.async_control.checkpoint_publish import _path_to_str
from vam.training.async_control.checkpoint_publish import _role_checkpoint_ref_is_complete
from vam.training.async_control.checkpoint_publish import _save_params
from vam.training.async_control.checkpoint_publish import _shape_of
from vam.training.async_control.checkpoint_publish import _tree_digest
from vam.training.async_control.checkpoint_publish import _unflatten_dict
from vam.training.async_control.dataset_view import DatasetView
from vam.training.async_control.dataset_view import DatasetViewWatcher
from vam.training.async_control.dataset_view import _episode_ids
from vam.training.async_control.dataset_view import _load_dataset_view_v1
from vam.training.async_control.dataset_view import _load_dataset_view_v2
from vam.training.async_control.dataset_view import _missing_feature_defaults
from vam.training.async_control.dataset_view import _repo_spec_defaults
from vam.training.async_control.dataset_view import apply_dataset_view
from vam.training.async_control.dataset_view import load_dataset_view
from vam.training.async_control.dataset_view import validate_stream_configs
from vam.training.async_control.train_state_switch import _copy_jax_array_buffers
from vam.training.async_control.train_state_switch import _count_like_step
from vam.training.async_control.train_state_switch import _put_like_reference
from vam.training.async_control.train_state_switch import _reset_optimizer_state_at_step
from vam.training.async_control.train_state_switch import _restore_params_from_params_dir
from vam.training.async_control.train_state_switch import _set_schedule_counts_to_step
from vam.training.async_control.train_state_switch import _switch_train_state_params
from vam.training.async_control.train_state_switch import _switch_train_state_params_impl
from vam.training.async_control.train_state_switch import _switch_train_state_to_current_ema_impl
from vam.training.async_control.train_state_switch import finalize_checkpoint_switch
from vam.training.async_control.train_state_switch import switch_train_state_checkpoint
from vam.training.async_control.train_state_switch import switch_train_state_to_current_ema
from vam.training.async_control.train_state_switch import validate_checkpoint_ref

__all__ = [
    "_VALUE_PARAM_PREFIXES",
    "CheckpointPublisher",
    "CheckpointSwitchWatcher",
    "DatasetView",
    "DatasetViewWatcher",
    "_atomic_write_json",
    "_checkpoint_dir_from_ref",
    "_copy_jax_array_buffers",
    "_copy_merged_assets",
    "_copytree_data_only",
    "_count_like_step",
    "_episode_ids",
    "_flatten_dict",
    "_load_dataset_view_v1",
    "_load_dataset_view_v2",
    "_matches_prefix",
    "_merge_value_params_into_action_params",
    "_merged_checkpoint_ref_is_complete",
    "_missing_feature_defaults",
    "_path_to_str",
    "_put_like_reference",
    "_read_json",
    "_repo_spec_defaults",
    "_reset_optimizer_state_at_step",
    "_resolve_path",
    "_restore_params_from_params_dir",
    "_role_checkpoint_ref_is_complete",
    "_save_params",
    "_set_schedule_counts_to_step",
    "_shape_of",
    "_switch_train_state_params",
    "_switch_train_state_params_impl",
    "_switch_train_state_to_current_ema_impl",
    "_tree_digest",
    "_unflatten_dict",
    "apply_dataset_view",
    "finalize_checkpoint_switch",
    "load_dataset_view",
    "switch_train_state_checkpoint",
    "switch_train_state_to_current_ema",
    "validate_checkpoint_ref",
    "validate_stream_configs",
]
