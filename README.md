# kist-vision-training

Converts kist-data-collector sessions (raw lossless recordings) into training
datasets.

## Requirements

- Python 3.10+ — one-liner setup:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

- System `ffmpeg` with libx264 (video encoding)
- No lerobot library needed — the converters write the LeRobot v2.1 files
  directly.

Input: a session directory containing `meta.yaml`, `lowstate.csv`,
`hand_{left,right}.csv`, `lowcmd.csv`, `hand_cmd_{left,right}.csv`, and
`head/color.h264 + color.idx.csv`.

## Usage

```bash
# GR00T N1.7 fine-tuning dataset (AppleToPlate schema) — the main converter
python3 src/preprocess/export_gr00t.py <sessions-root> --out <dataset-dir>

# e.g. success episodes only, ready for launch_finetune.py --dataset-path
python3 src/preprocess/export_gr00t.py ~/sessions_0818 --out ~/gr00t_train --results success
```

Options: `--results success,fail,unlabeled` (default `all`; label is kept in
`meta/episodes.jsonl` either way), `--teleop-only`, `--camera head`,
`--fps 30`, `--base-height 0.72`. The output dir must be empty.

`src/preprocess/export_lerobot.py` (same flags) produces the
Humanoid-Everyday-G1 schema instead — only needed when merging with that
dataset.

## Output structure

```
<dataset-dir>/
├── data/chunk-000/
│   └── episode_000000.parquet   # one row per 30 Hz timestep:
│                                #   observation.state[43], action[43],
│                                #   action.effort_* x7, navigate_command[3],
│                                #   base_height_command[1], timestamp, indices
├── videos/chunk-000/
│   └── observation.images.ego_view/
│       └── episode_000000.mp4   # head camera, h264, 30 fps, 480x640
└── meta/
    ├── info.json                # schema: features, fps, splits, path templates
    ├── modality.json            # GR00T group slices of the 43-dim vectors
    ├── tasks.jsonl              # task_index <-> language instruction
    ├── episodes.jsonl           # per-episode length, task, result, session
    └── episodes_stats.jsonl     # per-episode min/max/mean/std (normalization)
```

Schema is copied from [nvidia/GR00T-N1.7-AppleToPlate](https://huggingface.co/datasets/nvidia/GR00T-N1.7-AppleToPlate)
(the dataset used by NVIDIA's official G1 fine-tuning tutorial); only
difference is the video codec (h264 here, AV1 there — both decode fine).
