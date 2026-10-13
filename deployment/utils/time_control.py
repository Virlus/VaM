#!/usr/bin/env python3
"""
Precise time control utilities for device timing.
change precise_loop_timing to compatible with Windows and Linux.

Author: Jun Lv , JunjieXu
"""

import time
from typing import Callable
import sys
import ctypes


def precise_sleep(dt: float, slack_time: float = 0.001, time_func: Callable[[], float] = time.monotonic) -> None:
    """
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.

    Args:
        dt: Time to sleep in seconds
        slack_time: Time to spin instead of sleep (default: 0.001 seconds)
        time_func: Time function to use (default: time.monotonic)
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass


def precise_wait(t_end: float, slack_time: float = 0.001, time_func: Callable[[], float] = time.monotonic) -> None:
    """
    Wait until a specific end time using hybrid sleep/spin approach.

    Args:
        t_end: Target end time
        slack_time: Time to spin instead of sleep (default: 0.001 seconds)
        time_func: Time function to use (default: time.monotonic)
    """
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass


def precise_loop_timing(interval: float) -> Callable[[], None]:
    """
    Automatically select loop timing strategy based on platform:
    - Windows: absolute timing, timeBeginPeriod optimization, 2ms hybrid wait
    - Linux:  relative timing, 1ms hybrid wait
    """

    # Use high-precision clock source uniformly
    time_func = time.perf_counter

    # ============================================
    # Windows platform
    # ============================================
    if sys.platform == "win32":
        # Windows platform optimization
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

        # Initialize target time
        next_target = time_func() + interval

        def wait_for_next_windows():
            nonlocal next_target

            # 1. Calculate remaining time
            remain = next_target - time_func()

            # 2. Hybrid Wait - threshold 0.002s
            if remain > 0.002:
                time.sleep(remain - 0.002)
                remain = next_target - time_func()

            # 3. Busy Wait
            while remain > 0:
                remain = next_target - time_func()

            # 4. Increment absolute time (attempts to catch up even if stalled)
            next_target += interval

        return wait_for_next_windows

    # ============================================
    # Linux/Other platforms
    # ============================================
    else:
        # Default parameter slack_time of the first function
        slack_time = 0.001
        start_time = time_func()

        def wait_for_next_linux():
            nonlocal start_time

            # Calculate elapsed time
            elapsed = time_func() - start_time
            sleep_time = max(0, interval - elapsed)

            if sleep_time > 0:
                # Here implements the logic of precise_sleep in the original function
                # Sleep for most of the time first
                if sleep_time > slack_time:
                    time.sleep(sleep_time - slack_time)

                # Busy wait for remaining time
                while (time_func() - start_time) < interval:
                    pass

            # Reset start point to current time (relative timing logic, no catch-up)
            start_time = time_func()

        return wait_for_next_linux
