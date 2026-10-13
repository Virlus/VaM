"""
This library is a JAX implementation of Welford's algorithm,
which is an online and parallel algorithm for calculating variances.

Original NumPy implementation inspired by jvf/welford.
JAX conversion for VAM.
"""

from typing import Tuple

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Welford:
    """Accumulator object for Welford's online / parallel variance algorithm.

    Attributes:
        count (jnp.ndarray): The number of accumulated samples (scalar).
        mean (jnp.ndarray): Mean of the accumulated samples.
        m2 (jnp.ndarray): Sum of squares of differences from the current mean.
    """

    count: jnp.ndarray
    mean: jnp.ndarray
    m2: jnp.ndarray

    @classmethod
    def create(cls, shape: Tuple[int, ...], dtype=jnp.float32) -> "Welford":
        """Initialize an empty Welford accumulator.

        Args:
            shape: Shape of the data elements.
            dtype: Dtype of the statistics (default float32).
        """
        return cls(
            count=jnp.zeros((), dtype=dtype),
            mean=jnp.zeros(shape, dtype=dtype),
            m2=jnp.zeros(shape, dtype=dtype),
        )

    @classmethod
    def create_from_data(cls, elements: jnp.ndarray) -> "Welford":
        """Initialize with a batch of data samples.

        Args:
            elements: Array of shape (batch_size, ...).
        """
        count = jnp.array(elements.shape[0], dtype=elements.dtype)
        mean = jnp.mean(elements, axis=0)
        # m2 = sum((x - mean)^2)
        m2 = jnp.sum(jnp.square(elements - mean), axis=0)
        return cls(count=count, mean=mean, m2=m2)

    @property
    def var_s(self) -> jnp.ndarray:
        """Sample variance (ddof=1)."""
        # Return NaN if count <= 1
        return jnp.where(self.count > 1, self.m2 / (self.count - 1), jnp.full_like(self.mean, jnp.nan))

    @property
    def var_p(self) -> jnp.ndarray:
        """Population variance (ddof=0)."""
        # Return NaN if count <= 0 (though count should be >= 0)
        return jnp.where(self.count > 0, self.m2 / self.count, jnp.full_like(self.mean, jnp.nan))

    def add(self, element: jnp.ndarray) -> "Welford":
        """Add one data sample.

        Args:
            element: Data sample of shape `shape`.
        """
        count = self.count + 1
        delta = element - self.mean
        mean = self.mean + delta / count
        m2 = self.m2 + delta * (element - mean)
        return self.replace(count=count, mean=mean, m2=m2)

    def add_all(self, elements: jnp.ndarray) -> "Welford":
        """Add multiple data samples.

        Args:
            elements: Batch of samples of shape (batch_size, ...).
        """
        # It's more efficient to compute stats for the batch and merge
        other = Welford.create_from_data(elements)
        return self.merge(other)

    def merge(self, other: "Welford") -> "Welford":
        """Merge this accumulator with another one."""
        count = self.count + other.count

        delta = other.mean - self.mean
        delta2 = delta * delta

        # Use numerically stable merge update
        # Mean update
        mean = self.mean + delta * (other.count / count)

        # Clip the mean to 0.0 to prevent negative progress weights
        mean = jnp.clip(mean, 0.0)

        # M2 update
        m2 = self.m2 + other.m2 + delta2 * (self.count * other.count / count)

        return self.replace(count=count, mean=mean, m2=m2)
