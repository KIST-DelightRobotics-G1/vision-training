# kist-vision-training

Converts kist-data-collector sessions (raw lossless recordings) into training
datasets.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Also requires system `ffmpeg` with libx264. No lerobot library needed.

## Converters (src/preprocess/)

- `export_sonic.py` — **main converter**: GR00T N1.7 fine-tuning dataset for
  `--embodiment-tag UNITREE_G1_SONIC` (50 Hz motion-token grid, GR00T-WholeBodyControl format)
- `export_sonic_trim.py` — export_sonic + leading-idle trimming (displacement-based
  right-arm/hand motion-start detection, `--pad` context; `--no-trim` to disable the cut)
- `export_sonic_trim_uwb.py` — export_sonic_trim + `observation.uwb[3]` record-only
  column (nearest-joined UWB x,y,z; not in modality.json, so the GR00T loader ignores it)
- `export_lerobot.py` — Humanoid-Everyday-G1 schema, for merging with that dataset;
  also the shared alignment front-end imported by the others
- `dataset_viewer.py` / `view_parquet.py` — inspection: per-episode HTML viewer
  (video + labeled rows) and labeled parquet dump/compare

## Usage

```bash
python3 src/preprocess/export_sonic.py <sessions-root> --out <dataset-dir>

# typical: success episodes only, with wrist cameras
python3 src/preprocess/export_sonic.py ~/sessions_0818 --out ~/sonic_train \
    --results success --wrist-cameras
```

Options: `--results success,fail,unlabeled` (default `all`; the label is kept
in `meta/episodes.jsonl` either way), `--wrist-cameras`, `--fps 50`,
`--min-rows 25`. The output dir must be empty.

## Output structure

```
<dataset-dir>/
├── data/chunk-000/
│   └── episode_000000.parquet   # one row per 50 Hz motion-token tick:
│                                #   observation.state[43], action.motion_token[64],
│                                #   action.wbc[43], projected_gravity, root quat,
│                                #   teleop hand joints, timestamp, indices
├── videos/chunk-000/
│   ├── observation.images.ego_view/episode_000000.mp4      # h264, 50 fps
│   ├── observation.images.left_wrist/...                   # with --wrist-cameras
│   └── observation.images.right_wrist/...
└── meta/
    ├── info.json                # features, fps, splits, path templates
    ├── modality.json            # GR00T group slices (UNITREE_G1_SONIC contract)
    ├── tasks.jsonl              # task_index <-> language instruction
    ├── episodes.jsonl           # per-episode length, task, result, session
    └── episodes_stats.jsonl     # per-episode min/max/mean/std
```

Then fine-tune (in an Isaac-GR00T environment):

```bash
uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path <dataset-dir> --embodiment-tag UNITREE_G1_SONIC ...
```
