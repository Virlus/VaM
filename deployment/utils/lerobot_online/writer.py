"""Sequential multi-episode writer; callers supply complete, final online frames."""

from __future__ import annotations

import copy
import json
import sys
from importlib.metadata import distribution, version
from pathlib import Path

from utils.lerobot_conversion_logging import quiet_lerobot_writer

from .bundle import validate_repo
from .protocol import ProtocolError, validate_contract


@quiet_lerobot_writer
def write_role(root, episodes, config, role, contract, *, raw_advantage=False, progress=None):
    """Consume episode iterables one at a time, permitting overlapping value prefixes.

    Use at most one finalizer process; image writer threads/processes are explicit.
    No frame builder or default reward/value/advantage is hidden in this writer.
    """
    from utils.lerobot_recording import check_runtime

    if sys.version_info < (3, 11):
        raise RuntimeError("Online LeRobot writer requires Python >= 3.11")
    check_runtime()
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.common.datasets.video_utils import encode_video_frames

    if not raw_advantage:
        validate_contract(contract)
    installed = json.loads(distribution("lerobot").read_text("direct_url.json") or "{}")
    if installed.get("vcs_info", {}).get("commit_id") != contract["lerobot_revision"] or version("datasets") != "3.6.0":
        raise ProtocolError("Writer dependencies differ from the pinned contract")
    if CODEBASE_VERSION != "v2.1" or role not in {"action", "value"}:
        raise ProtocolError("Unsupported writer role/version")
    options = config["writer"]
    if options["image_writer_processes"] > 1 or not 1 <= options["image_writer_threads"] <= 8:
        raise ProtocolError("Online writer concurrency exceeds configured v1 bounds")
    frame_count = 0
    episode_number = 0
    total_episodes = len(episodes) if hasattr(episodes, "__len__") else "?"

    def report(stage):
        if progress:
            progress(f"{role} {episode_number}/{total_episodes} {stage}")

    class ConfiguredDataset(LeRobotDataset):
        def _wait_image_writer(self):
            report("png")
            return super()._wait_image_writer()

        def encode_episode_videos(self, episode_index):
            paths = {}
            for video_index, key in enumerate(self.meta.video_keys, 1):
                report(f"video {video_index}/{len(self.meta.video_keys)}")
                path = self.root / self.meta.get_video_file_path(episode_index, key)
                image_dir = self._get_image_file_path(episode_index, key, 0).parent
                for index in range(frame_count):
                    png = self._get_image_file_path(episode_index, key, index)
                    if not png.is_file() or png.stat().st_size == 0:
                        raise ProtocolError(f"Missing encoded input: {png}")
                encode_video_frames(image_dir, path, self.fps, overwrite=True, **options["video"])
                paths[key] = str(path)
            return paths

    features = copy.deepcopy(config["features"])
    for feature in features.values():
        feature["shape"] = tuple(feature["shape"])
    root = Path(root)
    dataset = ConfiguredDataset.create(
        repo_id=f"collection/{root.parent.name}_{role}",
        root=root,
        robot_type=config["robot_type"],
        fps=config["conversion"]["fps"],
        features=features,
        image_writer_processes=options["image_writer_processes"],
        image_writer_threads=options["image_writer_threads"],
        video_backend="pyav",
    )
    lengths = []
    try:
        for episode in episodes:
            episode_number += 1
            report("frames")
            if role == "action" and lengths:
                raise ProtocolError("Action role requires exactly one episode")
            frame_count = 0
            for frame in episode:
                dataset.add_frame(dict(frame))
                frame_count += 1
            if not frame_count:
                raise ProtocolError("Empty output episode")
            dataset.save_episode()
            lengths.append(frame_count)
        if not lengths:
            raise ProtocolError("Missing output episodes")
    finally:
        dataset.stop_image_writer()
    if progress:
        progress(f"{role} validating")
    validate_repo(root, lengths, role, contract, raw_advantage=raw_advantage)
    return {
        "path": role,
        "episode_count": len(lengths),
        "frame_count": sum(lengths),
        "episode_lengths": lengths,
    }
