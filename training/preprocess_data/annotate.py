"""Public entry point for model value predictions and derived advantage labels."""

import argparse
import importlib
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMMAND_MODULES = {
    "value": "preprocess_data.utils.value_annotation",
    "advantage": "preprocess_data.utils.advantage_annotation",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Let the selected command own its arguments and help, without loading both models.
    if argv and argv[0] in COMMAND_MODULES:
        return importlib.import_module(COMMAND_MODULES[argv[0]]).main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMAND_MODULES)
    parser.parse_args(argv)
    return None


if __name__ == "__main__":
    main()
