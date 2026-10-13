"""Read offline collect repos and compare local/cloud payload SHA256 inventories.

Run after the recorder has finished saving. Uses only LeRobot/data dependencies,
so the same file can run in the server training environment without this project.
"""

import argparse
import hashlib
import json
from pathlib import Path


def verify_collection(root, *, require_complete=False, expected=None):
    import av
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    episodes = sorted(p for p in root.glob("episode_*") if p.is_dir() and not p.name.endswith("_temp"))
    if not episodes:
        raise ValueError(f"No completed episodes in {root}")
    report = {"episodes": {}, "files": {}}
    for episode in episodes:
        if require_complete and not (episode / "complete.txt").is_file():
            raise ValueError(f"Upload not complete: {episode.name}")
        manifest_path = episode / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete" or not manifest.get("normal_end"):
            raise ValueError(f"Recording not complete: {episode.name}")
        if (episode / "cache").exists():
            raise ValueError(f"Wait for cache cleanup before verification: {episode}")
        report["episodes"][episode.name] = {}
        files = [manifest_path]
        for role in ("action", "value"):
            repo = episode / role
            info = json.loads((repo / "meta/info.json").read_text())
            parquet_files = sorted((repo / "data").rglob("*.parquet"))
            if info["codebase_version"] != "v2.1" or len(parquet_files) != 1:
                raise ValueError(f"Expected a single-episode LeRobot v2.1 repo: {repo}")
            table = pq.read_table(parquet_files[0])
            count = table.num_rows
            if (
                count == 0
                or count != info["total_frames"]
                or info["total_episodes"] != 1
                or count != manifest["output_frames"][role]
            ):
                raise ValueError(f"Frame counts disagree: {repo}")
            if not np.all(np.asarray(table["data_source"].to_pylist()) == 0):
                raise ValueError(f"Expected offline data_source=0: {repo}")
            if role == "action" and (
                "advantage" not in table.column_names
                or info["features"]["advantage"]["dtype"] != "float32"
                or info["features"]["advantage"]["shape"] != [1]
                or not np.all(np.asarray(table["advantage"].to_pylist()) == 1)
            ):
                raise ValueError(f"Expected local action advantage=1: {repo}")
            dataset = LeRobotDataset(f"collect/{episode.name}_{role}", root=repo, video_backend="pyav")
            for index in (0, count - 1):
                sample = dataset[index]
                if tuple(sample["actions"].shape) != tuple(info["features"]["actions"]["shape"]):
                    raise ValueError(f"Unexpected action shape: {repo}")
            # Decode every frame, including the six current/next image streams.
            for key in dataset.meta.video_keys:
                video = repo / dataset.meta.get_video_file_path(0, key)
                with av.open(str(video)) as container:
                    decoded = sum(1 for _ in container.decode(video=0))
                if decoded != count:
                    raise ValueError(f"Video frame count mismatch: {video}")
            report["episodes"][episode.name][role] = {
                "repo_path": repo.relative_to(root).as_posix(),
                "frames": count,
            }
            # LeRobot payload only; the remote completion marker is not source data.
            for directory in ("meta", "data", "videos"):
                files.extend(p for p in (repo / directory).rglob("*") if p.is_file())
        for path in sorted(files):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            report["files"][path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
    if expected is not None and report != expected:
        raise ValueError("Local/cloud payload reports differ (episodes, files or hashes)")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    expected = json.loads(args.expected.read_text()) if args.expected else None
    report = verify_collection(args.root, require_complete=args.require_complete, expected=expected)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {len(report['episodes'])} episodes, {len(report['files'])} files")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
