#!/usr/bin/env python3
"""Resolve the canonical offline repo from an async action dataset view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _resolve_view_path(latest_path: Path) -> Path:
    latest = _read_json(latest_path)
    view_ref = latest.get("view_path") or latest.get("dataset_view_path") or latest.get("path")
    if view_ref is None:
        return latest_path
    view_path = Path(str(view_ref)).expanduser()
    return view_path if view_path.is_absolute() else latest_path.parent / view_path


def _offline_repo_ids(view: dict[str, Any]) -> list[str]:
    repos = view.get("repos")
    if isinstance(repos, dict):
        return [
            str(repo_id)
            for repo_id, spec in repos.items()
            if float(((spec or {}).get("filters") or {}).get("data_source", -1)) == 0
        ]

    offline_repos = []
    for pool in (view.get("pools") or {}).values():
        if float(((pool or {}).get("filters") or {}).get("data_source", -1)) != 0:
            continue
        offline_repos.extend(str(source["repo_id"]) for source in (pool.get("sources") or []) if source.get("repo_id"))
    return offline_repos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("latest_path", type=Path, help="action_latest.json or a dataset_view.json")
    args = parser.parse_args()

    latest_path = args.latest_path.expanduser().resolve()
    view_path = _resolve_view_path(latest_path)
    offline_repos = sorted(set(_offline_repo_ids(_read_json(view_path))))
    if len(offline_repos) != 1:
        raise ValueError(
            f"Expected exactly one data_source=0 repo in {view_path}, got {offline_repos}. "
            "Set ALOHA_ASYNC_OFFLINE_REPO_ID explicitly when a view intentionally has multiple offline repos."
        )
    print(offline_repos[0])


if __name__ == "__main__":
    main()
