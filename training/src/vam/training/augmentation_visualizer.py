#!/usr/bin/env python3
"""
Data augmentation visualization utilities for training process.
Visualizes the augmentation coverage by applying augmentations multiple times to the same sample.
"""

import logging
import pathlib
from typing import Optional

import jax
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import vam.models.model as _model
from vam.shared import array_typing as at

logger = logging.getLogger(__name__)

# Global visualizer instance
_global_augmentation_visualizer = None


def convert_image_to_uint8(image: np.ndarray | jax.Array) -> np.ndarray:
    """Convert image from [-1, 1] or [0, 1] range to uint8 [0, 255]."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    # Check if image is in [-1, 1] range
    if image.min() < 0:
        image = (image + 1.0) / 2.0
    # Convert to [0, 255]
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def current_frame(image: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
    """Returns the current frame from a batched image or image-history tensor."""
    if len(image.shape) == 5:  # (batch, history, H, W, C)
        return image[0, -1]
    if len(image.shape) == 4:  # (batch, H, W, C)
        return image[0]
    return image


def initialize_augmentation_visualizer(
    num_augmentations: int = 20,
    save_interval: int = 1000,
    enabled: bool = True,
):
    """Initialize the global augmentation visualizer."""
    global _global_augmentation_visualizer
    if enabled:
        _global_augmentation_visualizer = AugmentationVisualizer(
            num_augmentations=num_augmentations,
            save_interval=save_interval,
        )
    else:
        _global_augmentation_visualizer = None


class AugmentationVisualizer:
    """Visualizes data augmentations applied during training."""

    def __init__(
        self,
        num_augmentations: int = 20,
        save_interval: int = 1000,
    ):
        """
        Initialize the augmentation visualizer.

        Args:
            num_augmentations: Number of augmentations to apply per sample
            save_interval: How often to save visualizations (in steps)
        """
        self.num_augmentations = num_augmentations
        self.save_interval = save_interval
        self.output_dir = None

    def set_output_dir(self, output_dir: str):
        """Set the output directory for visualizations."""
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_save_augmentations(self, step: int) -> bool:
        """Check if we should save augmentations at this step."""
        if self.output_dir is None:
            return False
        return step % self.save_interval == 0

    def save_augmented_images(
        self,
        original_obs: _model.Observation,
        augmented_obs_list: list[_model.Observation],
        step: int,
    ):
        """Save original and augmented images to disk."""
        if self.output_dir is None:
            return

        step_dir = self.output_dir / f"step_{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Get image keys
        image_keys = list(original_obs.images.keys())
        num_cameras = len(image_keys)

        # Convert original images
        original_images = {}
        for key in image_keys:
            img = current_frame(original_obs.images[key])
            original_images[key] = convert_image_to_uint8(jax.device_get(img))

        # Create a grid visualization
        num_aug = len(augmented_obs_list)
        fig, axes = plt.subplots(
            num_cameras + 1,  # +1 for original
            num_aug + 1,  # +1 for original column
            figsize=(1.5 * (num_aug + 1), 1.5 * (num_cameras + 1)),
        )

        # Plot original images in first column
        for cam_idx, key in enumerate(image_keys):
            axes[cam_idx, 0].imshow(original_images[key])
            axes[cam_idx, 0].set_title(f"Original\n{key}", fontsize=7)
            axes[cam_idx, 0].axis("off")

        # Plot augmented images
        for aug_idx, aug_obs in enumerate(augmented_obs_list):
            for cam_idx, key in enumerate(image_keys):
                img = current_frame(aug_obs.images[key])
                img_uint8 = convert_image_to_uint8(jax.device_get(img))

                axes[cam_idx, aug_idx + 1].imshow(img_uint8)
                axes[cam_idx, aug_idx + 1].set_title(f"#{aug_idx + 1}", fontsize=6)
                axes[cam_idx, aug_idx + 1].axis("off")

        # Hide the last row (empty)
        for col_idx in range(num_aug + 1):
            axes[-1, col_idx].axis("off")

        plt.tight_layout()
        plt.savefig(step_dir / "augmentation_grid.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved augmentation grid to {step_dir / 'augmentation_grid.png'}")

        # Save individual images
        individual_dir = step_dir / "individual_images"
        individual_dir.mkdir(exist_ok=True)

        # Save original images
        for key in image_keys:
            img = Image.fromarray(original_images[key])
            img.save(individual_dir / f"original_{key}.png")

        # Save augmented images
        for aug_idx, aug_obs in enumerate(augmented_obs_list):
            for key in image_keys:
                img = current_frame(aug_obs.images[key])
                img_uint8 = convert_image_to_uint8(jax.device_get(img))
                img_pil = Image.fromarray(img_uint8)
                img_pil.save(individual_dir / f"aug_{aug_idx + 1:03d}_{key}.png")

    def visualize_augmentations(
        self,
        observation: _model.Observation,
        step: int,
    ):
        """Visualize augmentations for a given observation."""
        if not self.should_save_augmentations(step):
            return

        # Extract first sample from batch
        with at.disable_typechecking():
            single_obs = _model.Observation(
                images={k: v[0:1] for k, v in observation.images.items()},  # Keep batch dim
                image_masks={k: v[0:1] for k, v in observation.image_masks.items()} if observation.image_masks else {},
                state=observation.state[0:1],
                state_mask=observation.state_mask[0:1] if observation.state_mask is not None else None,
                tokenized_prompt=observation.tokenized_prompt[0:1]
                if observation.tokenized_prompt is not None
                else None,
                tokenized_prompt_mask=observation.tokenized_prompt_mask[0:1]
                if observation.tokenized_prompt_mask is not None
                else None,
                tokenized_value_prompt=observation.tokenized_value_prompt[0:1]
                if observation.tokenized_value_prompt is not None
                else None,
                tokenized_value_prompt_mask=observation.tokenized_value_prompt_mask[0:1]
                if observation.tokenized_value_prompt_mask is not None
                else None,
                token_ar_mask=observation.token_ar_mask[0:1] if observation.token_ar_mask is not None else None,
                token_loss_mask=observation.token_loss_mask[0:1] if observation.token_loss_mask is not None else None,
                tokenized_advantage=observation.tokenized_advantage[0:1]
                if observation.tokenized_advantage is not None
                else None,
                tokenized_advantage_mask=observation.tokenized_advantage_mask[0:1]
                if observation.tokenized_advantage_mask is not None
                else None,
            )

        # Get original (non-augmented) observation for comparison
        original_obs = _model.preprocess_observation(
            None,  # No RNG means no augmentation
            single_obs,
            train=False,  # No augmentation
        )

        # Apply augmentations with different random seeds
        augmented_obs_list = []
        for aug_idx in range(self.num_augmentations):
            rng = jax.random.key(step * 1000 + aug_idx)  # Use step-based seed for reproducibility
            augmented_obs = _model.preprocess_observation(
                rng,
                single_obs,
                train=True,  # Enable augmentation
            )
            augmented_obs_list.append(augmented_obs)

        # Save visualizations
        self.save_augmented_images(original_obs, augmented_obs_list, step)


def maybe_visualize_augmentations(
    step: int,
    observation: _model.Observation,
    checkpoint_dir: Optional[str] = None,
    save_interval: int = 1000,
    num_augmentations: int = 20,
) -> bool:
    """
    Hook function to maybe visualize augmentations during training.

    This is a simple function that can be called from the training loop
    without major modifications to the training code.

    Args:
        step: Training step number
        observation: Observation data (before augmentation)
        checkpoint_dir: Checkpoint directory path (where to save visualizations)
        save_interval: How often to save (in steps)
        num_augmentations: Number of augmentations to apply

    Returns:
        True if visualization was saved, False otherwise
    """
    global _global_augmentation_visualizer

    if _global_augmentation_visualizer is None:
        # Initialize with default parameters if not already initialized
        initialize_augmentation_visualizer(
            num_augmentations=num_augmentations,
            save_interval=save_interval,
        )

    if _global_augmentation_visualizer is None:
        return False

    if checkpoint_dir is not None:
        _global_augmentation_visualizer.set_output_dir(pathlib.Path(checkpoint_dir) / "augmentation_visualizations")

    _global_augmentation_visualizer.visualize_augmentations(observation, step)
    return True
