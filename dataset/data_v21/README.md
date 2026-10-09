# cr3_o6 LeRobot v2.1 训练数据集（156 段）

由 `/home/ace/datasets/data`（合并后的采集器 v3.0 会话）转换而来，供
`/home/ace/cyf/openpi_cr3_o6` 的 OpenPI 训练使用。

## 结论

| 项目 | 值 |
| --- | --- |
| 目录 | `/home/ace/datasets/data_v21` |
| repo_id | `local/data_v21` |
| LeRobot 版本 | `codebase_version: v2.1` |
| 段数 | **156** |
| 帧数 | **84,710** |
| 视频文件 | 468（156 段 × 3 相机） |
| fps | 20 |
| 任务 | 单任务：`Pick up the motor and place it on the right side with the protruding side facing left.` |
| 机器人 | `robot_type: cr3_o6`（转换脚本写入值） |
| 特征 | `observation.state` / `action` 各 12 维 float32；三路 `observation.images.*_0_rgb` 224×224×3 H.264 |
| 大小 | 约 116 MB |

## 布局

```text
data_v21/
├── data/chunk-000/episode_000000.parquet …   # 156 个，每段一个，含 state/action/timestamp/frame_index/episode_index/index/task_index
├── videos/chunk-000/observation.images.<key>/episode_000000.mp4 …   # 468 个，每段每相机一个
├── meta/info.json                            # total_episodes=156 / total_frames=84710 / features
├── meta/episodes.jsonl                       # 156 行，逐段 tasks 与 length
├── meta/episodes_stats.jsonl                 # 156 行，逐段统计
├── meta/tasks.jsonl                          # 1 行
├── conversion.json                           # 来源会话、逐段 source→output 映射、排除列表（为空）
├── audit_verification.json                   # 校验结果与逐段来源对照
└── verify_v21.py                             # 复现校验
```

`physical state/action` 单位与来源一致：前 6 维为 CR3 关节角/绝对目标（弧度），后 6 维为
O6 位置/命令（0–255）。未做任何归一化，避免训练时重复归一化。

## 生成方式

```bash
cd /home/ace/cyf/openpi_cr3_o6
HF_HUB_OFFLINE=1 .venv/bin/python examples/cr3_o6/convert_data_to_lerobot.py \
  --source-roots /home/ace/datasets/data \
  --repo-id local/data_v21 \
  --output-root /home/ace/datasets/data_v21 \
  --allow-needs-review
```

`--allow-needs-review` 是必需的：采集器对 4 个会话的全部 156 段都写了
`quality.needs_review = true`（其固定标记），不加该参数转换脚本会拒绝执行。
所有 156 段本身都是 `outcome=success` 且 `recording_mode=joint`，因此**一段未排除**
（见 `conversion.json` 的 `excluded_episodes: []`）。

视频由解码后的 224×224 RGB 帧重新编码（`h264 / yuv420p / g=20 / crf=28`），
与你上次 `ceshi_merged_reviewed_20260912` 的转换参数一致。

## 校验

`audit_verification.json` 记录以下全部通过的检查（用 openpi 环境
`lerobot 0.1.0 (v2.1) + datasets 3.6.0` 实际执行）：

- `meta/info.json`：156 段、84,710 帧、单任务、468 视频、224×224。
- `episodes.jsonl` 156 行，序号 0–155 连续，长度之和等于总帧数。
- 每段恰好 1 个数据 parquet + 3 个视频。
- `LeRobotDataset` 实际加载：`len(dataset) = 84710`，与逐段 parquet 的值逐值比对，
  `observation.state` / `action` 与合并源完全相同。
- 抽检 9 段 × 3 帧，三路图像解码为 3×224×224 且数值有限，timestamp 为 `frame_index/20`。

已知差异（不影响训练）：

- `robot_type` 为 `cr3_o6`，而合并源 `data/meta/info.json` 为 `dobot_cr3_o6`
  （沿用转换脚本内的取值；OpenPI 训练不读取该字段）。
- v2.1 的 `features.*.info.video.height/width` 仍写 480/640（转换脚本从来源特征复制），
  而实际视频是 224×224。OpenPI 读取时按图像张量形状处理，不受该字段影响。

## 接入训练

训练读取的是 `HF_LEROBOT_HOME / repo_id`。本机默认缓存目录需建立映射：

```bash
ln -s /home/ace/datasets/data_v21 ~/.cache/huggingface/lerobot/local/data_v21
```

或者把 `HF_LEROBOT_HOME` 指到自建目录（该目录下同样需要 `local/data_v21` 指向本数据集），
例如 `/home/ace/datasets/lerobot_cache/local/data_v21`。

之后在 `src/openpi/training/config.py` 中新配置的 `repo_id` 写 `local/data_v21` 即可；
若要跑归一化统计，可用 `examples/cr3_o6/compute_joint_norm_stats.py`
（它同样按 `HF_LEROBOT_HOME / repo_id` 解析路径）。

## 复现校验

```bash
cd /home/ace/datasets
/home/ace/cyf/openpi_cr3_o6/.venv/bin/python verify_v21.py     # 期望输出 V21_VERIFY_OK
```
