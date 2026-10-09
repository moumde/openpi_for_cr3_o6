"""Verify the converted LeRobot v2.1 dataset with the OpenPI training environment's lerobot.

Usage:
    HF_HUB_OFFLINE=1 /home/ace/cyf/openpi_cr3_o6/.venv/bin/python verify_v21.py \
        [--root /home/ace/datasets/data_v21] [--repo-id local/data_v21] [--merged /home/ace/datasets/data]

Checks:
  * metadata: repo id, fps, single task, 156 episodes, 84710 frames, feature names/shapes
  * episodes.jsonl: one entry per episode, single task each, lengths sum to the frame total
  * per episode: exactly one data parquet and one video per camera
  * real reads through lerobot.LeRobotDataset (metadata, tasks, stats, video decode)
  * sample frames decode at the declared resolution and match nothing-but-shape expectations
  * state/action/timestamp agree with the merged v3.0 source for the sampled episodes
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# The training environment's `datasets` build writes a lock/builder cache; keep it inside the
# workspace so verification does not depend on the read-only ~/.cache/huggingface tree.
_CACHE = Path("/home/ace/datasets/.hf_cache")
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pyarrow.parquet as pq
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

KEY_TO_V3 = {
    "observation.images.base_0_rgb": "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb": "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb": "observation.images.right_wrist_0_rgb",
}
SAMPLE_EPISODES = 8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/ace/datasets/data_v21"))
    parser.add_argument("--repo-id", default="local/data_v21")
    parser.add_argument("--merged", type=Path, default=Path("/home/ace/datasets/data"))
    parser.add_argument("--expected-episodes", type=int, default=156)
    args = parser.parse_args()

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)
            print(f"FAIL {message}")

    info = json.loads((args.root / "meta/info.json").read_text())
    print(f"info.json: version={info['codebase_version']} fps={info['fps']} "
          f"episodes={info['total_episodes']} frames={info['total_frames']} tasks={info['total_tasks']} "
          f"videos={info['total_videos']} chunks={info['total_chunks']}")
    if "repo_id" in info:
        check(info["repo_id"] == args.repo_id, f"info repo_id is {info['repo_id']}")
    check(info["total_episodes"] == args.expected_episodes,
          f"info total_episodes is {info['total_episodes']}, expected {args.expected_episodes}")
    check(info["fps"] == 20, "fps is not 20")
    check(info["total_tasks"] == 1, "expected exactly one task")
    check(info["codebase_version"] == "v2.1", f"codebase_version is {info['codebase_version']}")

    merged_info = json.loads((args.merged / "meta/info.json").read_text())
    check(info["total_frames"] == merged_info["total_frames"],
          f"total_frames {info['total_frames']} != merged {merged_info['total_frames']}")
    check(info["total_episodes"] == merged_info["total_episodes"], "total_episodes != merged")
    for key, feature in merged_info["features"].items():
        if feature["dtype"] != "video":
            continue
        check(key in info["features"], f"missing video feature {key}")
        check(info["features"][key]["shape"][:2] == [224, 224],
              f"{key} shape is {info['features'][key]['shape']}, expected 224x224")

    episodes = [json.loads(line) for line in (args.root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    tasks = [json.loads(line) for line in (args.root / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
    stats_lines = [json.loads(line) for line in (args.root / "meta/episodes_stats.jsonl").read_text().splitlines() if line.strip()]
    print(f"episodes.jsonl: {len(episodes)} entries, tasks.jsonl: {len(tasks)}, episodes_stats.jsonl: {len(stats_lines)}")
    check(len(episodes) == args.expected_episodes, f"episodes.jsonl has {len(episodes)} entries")
    check(len(tasks) == 1, f"tasks.jsonl has {len(tasks)} entries")
    check(len(stats_lines) == args.expected_episodes, "episodes_stats.jsonl count mismatch")
    check([e["episode_index"] for e in episodes] == list(range(args.expected_episodes)),
          "episode indices are not 0..N-1 in order")
    check(sum(e["length"] for e in episodes) == info["total_frames"],
          "episode lengths do not sum to total_frames")
    check(all(len(e["tasks"]) == 1 and e["tasks"][0] == tasks[0]["task"] for e in episodes),
          "episodes do not all carry the single task")

    # one parquet and one video per camera per episode
    data_files = sorted((args.root / "data").rglob("*.parquet"))
    check(len(data_files) == args.expected_episodes,
          f"{len(data_files)} data parquet files, expected {args.expected_episodes}")
    for key in KEY_TO_V3:
        camera_dir = args.root / "videos" / "chunk-000" / key
        video_files = sorted(camera_dir.rglob("*.mp4"))
        check(len(video_files) == args.expected_episodes,
              f"{key}: {len(video_files)} video files, expected {args.expected_episodes}")

    # real reads through the training environment's lerobot
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    dataset = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav")
    print(f"lerobot: len(dataset)={len(dataset)}  meta episodes={metadata.total_episodes} "
          f"frames={metadata.total_frames}  tasks={metadata.tasks}")
    check(len(dataset) == info["total_frames"], "len(dataset) != total_frames")
    check(metadata.total_episodes == args.expected_episodes, "metadata total_episodes mismatch")
    check(metadata.total_frames == info["total_frames"], "metadata total_frames mismatch")
    check(len(metadata.tasks) == 1, "metadata tasks count mismatch")

    merged_data = pq.read_table(args.merged / "data/chunk-000/file-000.parquet",
                               columns=["observation.state", "action", "timestamp",
                                        "episode_index", "frame_index"])
    merged_by_episode: dict[int, dict] = {}
    for row in merged_data.to_pylist():
        merged_by_episode.setdefault(row["episode_index"], {})[row["frame_index"]] = row

    # global frame offset of each episode in the v2.1 dataset
    episode_offsets = np.cumsum([0] + [e["length"] for e in episodes[:-1]])
    step = max(1, args.expected_episodes // SAMPLE_EPISODES)
    sampled = 0
    for episode in episodes[::step]:
        episode_index = episode["episode_index"]
        length = episode["length"]
        base = int(episode_offsets[episode_index])
        for frame_index in (0, length // 2, length - 1):
            item = dataset[base + frame_index]
            check(tuple(item["observation.state"].shape) == (12,),
                  f"ep {episode_index} frame {frame_index}: state shape {tuple(item['observation.state'].shape)}")
            check(tuple(item["action"].shape) == (12,),
                  f"ep {episode_index} frame {frame_index}: action shape {tuple(item['action'].shape)}")
            check(float(item["timestamp"]) == pytest_approx(frame_index / 20.0, 1e-4),
                  f"ep {episode_index} frame {frame_index}: timestamp {float(item['timestamp'])}")
            for key in KEY_TO_V3:
                image = item[key]
                check(tuple(image.shape) == (3, 224, 224),
                      f"ep {episode_index} frame {frame_index}: {key} shape {tuple(image.shape)}")
                check(bool(np.isfinite(np.asarray(image)).all()),
                      f"ep {episode_index} frame {frame_index}: {key} has non-finite values")
            source = merged_by_episode[episode_index][frame_index]
            check(np.allclose(np.asarray(item["observation.state"], dtype=np.float32),
                              np.asarray(source["observation.state"], dtype=np.float32), atol=0, rtol=0),
                  f"ep {episode_index} frame {frame_index}: state differs from merged source")
            check(np.allclose(np.asarray(item["action"], dtype=np.float32),
                              np.asarray(source["action"], dtype=np.float32), atol=0, rtol=0),
                  f"ep {episode_index} frame {frame_index}: action differs from merged source")
            sampled += 1
    print(f"sampled {sampled} frames across {len(episodes[::step])} episodes via lerobot")

    print()
    if failures:
        print(f"V21_VERIFY_FAILED {len(failures)} problem(s)")
        return 1
    print("V21_VERIFY_OK")
    return 0


def pytest_approx(value: float, tol: float):
    class _Approx:
        def __init__(self, v, t):
            self.v, self.t = v, t

        def __eq__(self, other):
            return abs(float(other) - self.v) <= self.t

        def __repr__(self):
            return f"{self.v}+-{self.t}"
    return _Approx(value, tol)


if __name__ == "__main__":
    raise SystemExit(main())
