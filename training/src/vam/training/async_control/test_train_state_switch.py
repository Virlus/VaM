from unittest import mock

import jax
import jax.numpy as jnp

from vam.training.async_control import train_state_switch


def test_finalize_checkpoint_switch_reclaims_python_memory_without_clearing_jax_caches() -> None:
    state = object()

    with (
        mock.patch.object(train_state_switch.jax, "block_until_ready") as block_until_ready,
        mock.patch.object(train_state_switch.gc, "collect") as collect,
        mock.patch.object(train_state_switch.jax, "clear_caches") as clear_caches,
    ):
        result = train_state_switch.finalize_checkpoint_switch(
            state,
            context="test handoff",
            reclaim_memory=True,
        )

    assert result is state
    block_until_ready.assert_called_once_with(state)
    collect.assert_called_once_with()
    clear_caches.assert_not_called()


def test_finalize_checkpoint_switch_can_explicitly_clear_jax_caches() -> None:
    state = object()

    with (
        mock.patch.object(train_state_switch.jax, "block_until_ready"),
        mock.patch.object(train_state_switch.gc, "collect") as collect,
        mock.patch.object(train_state_switch.jax, "clear_caches") as clear_caches,
    ):
        train_state_switch.finalize_checkpoint_switch(
            state,
            context="oom recovery",
            clear_jax_caches=True,
        )

    collect.assert_not_called()
    clear_caches.assert_called_once_with()


def test_finalize_checkpoint_switch_can_reclaim_python_memory_and_clear_jax_caches() -> None:
    state = object()

    with (
        mock.patch.object(train_state_switch.jax, "block_until_ready"),
        mock.patch.object(train_state_switch.gc, "collect") as collect,
        mock.patch.object(train_state_switch.jax, "clear_caches") as clear_caches,
    ):
        train_state_switch.finalize_checkpoint_switch(
            state,
            context="oom recovery",
            reclaim_memory=True,
            clear_jax_caches=True,
        )

    collect.assert_called_once_with()
    clear_caches.assert_called_once_with()


def test_repeated_checkpoint_switches_retain_compiled_executable_across_multiple_steps() -> None:
    trace_count = 0

    @jax.jit
    def train_step(value):
        nonlocal trace_count
        trace_count += 1
        return value + 1

    value = jnp.asarray(0)
    value = train_step(value)
    for handoff in range(3):
        train_state_switch.finalize_checkpoint_switch(
            value,
            context=f"test handoff {handoff}",
            reclaim_memory=True,
        )
        for _ in range(5):
            value = train_step(value)
    jax.block_until_ready(value)

    assert trace_count == 1
