#!/usr/bin/env python3
"""Data collection: record paired LeRobot datasets from the configured I2RT streams."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from consumer.lerobot_data_saver import main

if __name__ == "__main__":
    main()
