import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch

from vam.shared import image_tools
import vam.shared.array_typing as at

logger = logging.getLogger("vam")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"
    PI0_SUFFIX_VALUE = "pi0_suffix_value"
    PI0_PREFIX_VALUE = "pi0_prefix_value"
    PI0_VALUE_EXPERT = "pi0_value_expert"
    PI0_Q_FUNCTION = "pi0_q_function"
    PI05_SUFFIX_VALUE = "pi05_suffix_value"
    PI05_PREFIX_VALUE = "pi05_prefix_value"
    PI05_VALUE_EXPERT = "pi05_value_expert"
    PI05_Q_FUNCTION = "pi05_q_function"
    PI06 = "pi06"


class TrainingType(enum.Enum):
    BC = "bc"
    TD_LEARNING = "td_learning"
    VALUE_MC_LEARNING = "value_mc_learning"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


class RandomTranslate(augmax.GeometricTransformation):
    """Apply a small random offset to an image using augmax's geometric API."""

    def __init__(self, max_offset_frac_y: float = 0.04, max_offset_frac_x: float | None = None):
        super().__init__()
        self._max_offset_frac_y = max_offset_frac_y
        self._max_offset_frac_x = max_offset_frac_x if max_offset_frac_x is not None else max_offset_frac_y

    def transform_coordinates(
        self,
        rng: jnp.ndarray,
        coordinates: "augmax.geometric.LazyCoordinates",
        invert=False,  # noqa: FBT002
    ):
        height, width = coordinates.current_shape

        max_dy = height * self._max_offset_frac_y
        max_dx = width * self._max_offset_frac_x

        dy = jax.random.uniform(rng, minval=-max_dy, maxval=max_dy)
        dx = jax.random.uniform(rng, minval=-max_dx, maxval=max_dx)

        if invert:
            dy = -dy
            dx = -dx

        # Augmax uses inverse mapping from destination to source pixels.
        transform = jnp.array(
            [
                [1, 0, -dx],
                [0, 1, -dy],
                [0, 0, 1],
            ]
        )
        coordinates.push_transform(transform)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*bh h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*bh"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*bh s"]
    # Optional state history masks, true where state entries are valid.
    state_mask: at.Bool[ArrayT, "*bh"] | None = None
    # Optional per-frame value predictions used as action-prefix conditioning.
    value_prediction: at.Float[ArrayT, "*bh 1"] | None = None
    value_prediction_mask: at.Bool[ArrayT, "*bh"] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # Tokenized value prompt.
    tokenized_value_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized value prompt mask.
    tokenized_value_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi06 model specific fields
    tokenized_advantage: at.Int[ArrayT, "*b adv"] | None = None
    tokenized_advantage_mask: at.Bool[ArrayT, "*b adv"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        if ("tokenized_advantage" in data) != ("tokenized_advantage_mask" in data):
            raise ValueError("tokenized_advantage and tokenized_advantage_mask must be provided together.")
        if ("tokenized_value_prompt" in data) != ("tokenized_value_prompt_mask" in data):
            raise ValueError("tokenized_value_prompt and tokenized_value_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32) / 255.0 * 2.0 - 1.0
        with at.disable_typechecking():
            return cls(
                images=data["image"],
                image_masks=data["image_mask"],
                state=data["state"],
                state_mask=data.get("state_mask"),
                value_prediction=data.get("value_prediction"),
                value_prediction_mask=data.get("value_prediction_mask"),
                tokenized_prompt=data.get("tokenized_prompt"),
                tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
                tokenized_value_prompt=data.get("tokenized_value_prompt"),
                tokenized_value_prompt_mask=data.get("tokenized_value_prompt_mask"),
                token_ar_mask=data.get("token_ar_mask"),
                token_loss_mask=data.get("token_loss_mask"),
                tokenized_advantage=data.get("tokenized_advantage"),
                tokenized_advantage_mask=data.get("tokenized_advantage_mask"),
            )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]
Values = at.Float[ArrayT, "*b 1"] | at.Float[ArrayT, "*b v"]
Success = at.Float[ArrayT, "*b 1"]
Reward = at.Float[ArrayT, "*b 1"] | at.Float[ArrayT, "*b ah"]
ControlFlag = at.Float[ArrayT, "*b 1"]
DataSource = at.Float[ArrayT, "*b 1"]
Terminate = at.Float[ArrayT, "*b 1"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[-3:-1] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[-3:-1]} to {image_resolution}")
            leading_shape = image.shape[:-3]
            image = image.reshape((-1, *image.shape[-3:]))
            image = image_tools.resize_with_pad(image, *image_resolution)
            image = image.reshape((*leading_shape, *image.shape[-3:]))

        if train:
            leading_shape = image.shape[:-3]
            image = image.reshape((-1, *image.shape[-3:]))
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            else:
                height, width = image.shape[1:3]
                transforms += [
                    RandomTranslate(max_offset_frac_y=0.025, max_offset_frac_x=0.025),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            if len(leading_shape) > 1:
                # Keep temporal views of the same sample under the same stochastic augmentation.
                # This preserves current-frame equivalence when a branch slices history to horizon 1.
                per_sample_rngs = jax.random.split(rng, leading_shape[0])
                sub_rngs = jnp.repeat(per_sample_rngs, int(np.prod(leading_shape[1:])), axis=0)
            else:
                sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0
            image = image.reshape((*leading_shape, *image.shape[-3:]))

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    with at.disable_typechecking():
        return Observation(
            images=out_images,
            image_masks=out_masks,
            state=observation.state,
            state_mask=observation.state_mask,
            value_prediction=observation.value_prediction,
            value_prediction_mask=observation.value_prediction_mask,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
            tokenized_value_prompt=observation.tokenized_value_prompt,
            tokenized_value_prompt_mask=observation.tokenized_value_prompt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
            tokenized_advantage=observation.tokenized_advantage,
            tokenized_advantage_mask=observation.tokenized_advantage_mask,
        )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        try:
            at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        except ValueError as e:
            logging.warning(f"Weights validation warning: {e}")
            logging.warning("Proceeding with partial loading.")
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        value: Values | None = None,
        next_obs: Observation | None = None,
        success: Success | None = None,
        reward: Reward | None = None,
        next_actions: Actions | None = None,
        control_flag: ControlFlag | None = None,
        data_source: DataSource | None = None,
        terminate: Terminate | None = None,
        *,
        train: bool = False,
        train_type: set[TrainingType] | None = None,
        filter_action_loss_by_success: bool = True,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...

    @abc.abstractmethod
    def predict_value(
        self,
        observation: Observation,
        actions: Actions | None = None,
    ) -> Values: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during VAM training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during VAM training, many key paths end with the "value" suffix added
    # by `nnx.State`. Strip the suffix wherever it appears so the tree matches `state.to_pure_dict()`.
    flat_params = traverse_util.flatten_dict(params)
    flat_params = {kp[:-1] if kp and kp[-1] == "value" else kp: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
