#!/usr/bin/env python3
"""kist-data-collector session(s) -> LeRobot v2.1 dataset (HE G1-lite schema).

Converts raw recorder sessions (blob+CSV, native rates) into the exact
schema of USC-PSI-Lab/Humanoid-Everyday-G1 so the two datasets can be
merged for GR00T fine-tuning:

  data/chunk-XXX/episode_XXXXXX.parquet   30 Hz rows (joints, action, indices)
  videos/chunk-XXX/egocentric/episode_XXXXXX.mp4
  meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl, stats.json}

Timestep = one decoded head color frame. High-rate streams attach per frame:
observations by nearest recv_ns (lowstate 1 kHz -> max +-0.5 ms error),
actions by backward recv_ns (the command standing at frame time).

  action (28) = hand_cmd_left f0..f6 q  (7)      # same layout as HE:
              + hand_cmd_right f0..f6 q (7)      #   [left hand, right hand,
              + lowcmd m15..m28 q       (14)     #    arm targets]

Episode label filter: each session's meta.yaml carries `result:`
(success / fail / "" = unlabeled, set by the operator at Ctrl-C).
--results picks which labels to export; every exported episode keeps its
label in episodes.jsonl (`result`) so training-side filtering stays possible.

Usage:
  python export_lerobot.py SESSION_DIR [SESSION_DIR ...] --out DATASET_DIR
  python export_lerobot.py SESSIONS_ROOT --out DATASET_DIR --results success
Options: --results all|success|fail|unlabeled (comma list, default all),
  --teleop-only (keep only arbiter_mode==1 spans), --camera head, --fps 30.
"""

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ARM_MOTORS = [f"m{i:02d}" for i in range(15, 29)]   # G1: 15..28 = arms (L then R)
LEG_MOTORS = [f"m{i:02d}" for i in range(0, 15)]    # 0..11 legs + 12..14 waist
HAND_MOTORS = [f"f{i}" for i in range(7)]           # Dex3 SDK order (thumb, middle, index)

FLOATLIST_FEATURES = [
    "observation.arm_joints", "observation.leg_joints",
    "observation.hand_joints", "action",
]


# ---------------------------------------------------------------- sessions

def parse_meta(session: Path) -> dict:
    """task / result out of meta.yaml (regex — the format: block is prose
    and not worth a strict-YAML dependency)."""
    text = (session / "meta.yaml").read_text(encoding="utf-8")
    task = re.search(r'^\s*task:\s*"(.*)"\s*$', text, re.M)
    result = re.search(r"^result:\s*(\S*)\s*$", text, re.M)
    return {
        "task": task.group(1) if task else "",
        "result": (result.group(1) if result else "").strip('"'),
    }


def discover_sessions(paths: list) -> list:
    """Each arg: a session dir (has meta.yaml) or a root of session dirs."""
    out = []
    for p in paths:
        d = Path(p).expanduser()
        if (d / "meta.yaml").is_file():
            out.append(d)
        elif d.is_dir():
            out.extend(sorted(c for c in d.iterdir() if (c / "meta.yaml").is_file()))
        else:
            sys.exit(f"error: not a session dir or root: {d}")
    if not out:
        sys.exit("error: no sessions found (dirs must contain meta.yaml)")
    return out


# ------------------------------------------------------------------ video

def decode_seqs(camera_dir: Path) -> list:
    """Pass 1: which frames actually decode (packets are fed with pts=seq, so
    the decoder itself reports each output frame's seq — post-gap undecodables
    and pre-keyframe frames simply never come out)."""
    return [seq for seq, _ in _decode(camera_dir)]


def _decode(camera_dir: Path):
    # Frames before the first keyframe are undecodable (no SPS/PPS yet) and
    # the strict send-packet API errors on them — start at the first IDR,
    # and tolerate per-packet errors right after wire gaps.
    idx = pd.read_csv(camera_dir / "color.idx.csv")
    idx = idx[idx["is_keyframe"].cumsum() > 0]
    blob = (camera_dir / "color.h264").open("rb")
    ctx = av.CodecContext.create("h264", "r")

    def emit(frames):
        for f in frames:
            if f.pts is not None:
                yield int(f.pts), f

    for row in idx.itertuples():
        blob.seek(row.offset)
        pkt = av.Packet(blob.read(row.size))
        pkt.pts = int(row.seq)
        try:
            yield from emit(ctx.decode(pkt))
        except av.error.InvalidDataError:
            continue
    yield from emit(ctx.decode(None))  # flush
    blob.close()


def write_video(camera_dir: Path, keep: set, out_path: Path, fps: int) -> int:
    """Pass 2: re-decode, pipe kept frames (decode order) into libx264 mp4.
    Returns frames written — must equal the parquet row count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = None
    written = 0
    for seq, frame in _decode(camera_dir):
        if seq not in keep:
            continue
        img = frame.to_ndarray(format="bgr24")
        if proc is None:
            h, w = img.shape[:2]
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", str(fps), "-i", "pipe:0",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                 str(out_path)],
                stdin=subprocess.PIPE)
        proc.stdin.write(img.tobytes())
        written += 1
    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit(f"error: ffmpeg failed for {out_path}")
    return written


# ------------------------------------------------------------------- rows

def _asof(frames: pd.DataFrame, csv: Path, cols: list, direction: str) -> pd.DataFrame:
    src = pd.read_csv(csv, usecols=["recv_ns"] + cols)
    src["recv_ns"] = src["recv_ns"].astype("int64")
    return pd.merge_asof(frames, src.sort_values("recv_ns"),
                         on="recv_ns", direction=direction)


def build_rows(session: Path, camera: str, decoded: list, teleop_only: bool):
    """One row per decodable frame; joins by recv_ns. Returns (df, kept_seqs)
    with flat per-dim columns (vectors are assembled in make_parquet)."""
    idx = pd.read_csv(session / camera / "color.idx.csv",
                      usecols=["seq", "recv_ns"]).astype("int64")
    frames = idx[idx["seq"].isin(decoded)].sort_values("recv_ns").reset_index(drop=True)

    obs_specs = [
        (session / "lowstate.csv", [f"{m}_q" for m in ARM_MOTORS + LEG_MOTORS], "st"),
        (session / "hand_left.csv", [f"{f}_q" for f in HAND_MOTORS], "hl"),
        (session / "hand_right.csv", [f"{f}_q" for f in HAND_MOTORS], "hr"),
    ]
    act_specs = [
        (session / "hand_cmd_left.csv", [f"{f}_q" for f in HAND_MOTORS], "cl"),
        (session / "hand_cmd_right.csv", [f"{f}_q" for f in HAND_MOTORS], "cr"),
        (session / "lowcmd.csv", [f"{m}_q" for m in ARM_MOTORS], "ca"),
    ]
    df = frames
    for csv, cols, tag in obs_specs:
        df = _asof(df, csv, cols, "nearest").rename(columns={c: f"{tag}.{c}" for c in cols})
    for csv, cols, tag in act_specs:
        # backward = the command standing at frame time, not a future one
        df = _asof(df, csv, cols, "backward").rename(columns={c: f"{tag}.{c}" for c in cols})

    if teleop_only:
        tok = pd.read_csv(session / "motion_token.csv",
                          usecols=["recv_ns", "arbiter_mode"])
        tok = tok[tok["arbiter_mode"] == 1].astype({"recv_ns": "int64"})
        if tok.empty:
            return df.iloc[0:0], []
        near = pd.merge_asof(df[["recv_ns"]], tok.sort_values("recv_ns"),
                             on="recv_ns", direction="nearest",
                             tolerance=40_000_000)  # 40 ms ~ 2 token periods @50Hz
        df = df[near["arbiter_mode"].notna().values]

    df = df.dropna()  # frames before the first command carry no action
    return df.reset_index(drop=True), df["seq"].astype(int).tolist()


def make_parquet(df: pd.DataFrame, ep_index: int, task_index: int,
                 global_start: int, out_path: Path, fps: int) -> dict:
    n = len(df)
    vec = {
        "observation.arm_joints": [f"st.{m}_q" for m in ARM_MOTORS],
        "observation.leg_joints": [f"st.{m}_q" for m in LEG_MOTORS],
        "observation.hand_joints": [f"hl.{f}_q" for f in HAND_MOTORS]
                                 + [f"hr.{f}_q" for f in HAND_MOTORS],
        "action": [f"cl.{f}_q" for f in HAND_MOTORS]
                + [f"cr.{f}_q" for f in HAND_MOTORS]
                + [f"ca.{m}_q" for m in ARM_MOTORS],
    }
    arrays, stats = {}, {}
    for feat, cols in vec.items():
        m = df[cols].to_numpy(dtype=np.float32)
        arrays[feat] = m
        stats[feat] = {
            "min": m.min(0).tolist(), "max": m.max(0).tolist(),
            "mean": m.mean(0).tolist(), "std": m.std(0).tolist(), "count": [n],
        }

    cols = {f: pa.array(list(arrays[f]), type=pa.list_(pa.float32()))
            for f in FLOATLIST_FEATURES}
    cols["timestamp"] = pa.array((np.arange(n) / fps).astype(np.float32),
                                 type=pa.float32())
    cols["frame_index"] = pa.array(np.arange(n), type=pa.int64())
    cols["episode_index"] = pa.array(np.full(n, ep_index), type=pa.int64())
    cols["index"] = pa.array(np.arange(global_start, global_start + n), type=pa.int64())
    cols["next.done"] = pa.array([i == n - 1 for i in range(n)], type=pa.bool_())
    cols["task_index"] = pa.array(np.full(n, task_index), type=pa.int64())

    order = FLOATLIST_FEATURES + ["timestamp", "frame_index", "episode_index",
                                  "index", "next.done", "task_index"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: cols[k] for k in order}), out_path)
    return stats


# ------------------------------------------------------------------- meta

def write_meta(out: Path, episodes: list, tasks: list, fps: int, chunks_size: int):
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    total_frames = sum(e["length"] for e in episodes)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "g1",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episodes),
        "total_chunks": math.ceil(len(episodes) / chunks_size),
        "chunks_size": chunks_size,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.egocentric": {
                "dtype": "video", "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": float(fps), "video.codec": "h264",
                               "video.pix_fmt": "yuv420p",
                               "video.is_depth_map": False, "has_audio": False},
            },
            **{f: {"dtype": "float32", "shape": [-1]} for f in FLOATLIST_FEATURES},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=4))

    # GR00T loader's schema map: which slice of which parquet column each
    # state/action group lives in. Plain LeRobot/HE tooling ignores this file;
    # only GR00T reads it. start/end index within the original_key column.
    modality = {
        "state": {
            "arm_joints":  {"start": 0, "end": 14, "original_key": "observation.arm_joints"},
            "leg_joints":  {"start": 0, "end": 15, "original_key": "observation.leg_joints"},
            "hand_joints": {"start": 0, "end": 14, "original_key": "observation.hand_joints"},
        },
        "action": {  # slices of the single `action` column (28-dim)
            "left_hand":  {"start": 0,  "end": 7},
            "right_hand": {"start": 7,  "end": 14},
            "arms":       {"start": 14, "end": 28},
        },
        "video": {
            "egocentric": {"original_key": "observation.images.egocentric"},
        },
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=4))

    with open(meta / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    with open(meta / "episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps({k: v for k, v in e.items() if k != "stats"}) + "\n")
    with open(meta / "episodes_stats.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps({"episode_index": e["episode_index"],
                                "stats": e["stats"]}) + "\n")

    # global action stats (HE's stats.json shape), aggregated over episodes
    a = [e["stats"]["action"] for e in episodes]
    counts = np.array([s["count"][0] for s in a])[:, None]
    means = np.array([s["mean"] for s in a])
    stds = np.array([s["std"] for s in a])
    gmean = (means * counts).sum(0) / counts.sum()
    gvar = ((counts * (stds**2 + means**2)).sum(0) / counts.sum()) - gmean**2
    (meta / "stats.json").write_text(json.dumps({"g1": {
        "action_min": np.min([s["min"] for s in a], 0).tolist(),
        "action_max": np.max([s["max"] for s in a], 0).tolist(),
        "action_mean": gmean.tolist(),
        "action_std": np.sqrt(np.maximum(gvar, 0)).tolist(),
    }}, indent=4))


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="kist session(s) -> LeRobot v2.1 (HE G1-lite schema)")
    ap.add_argument("sessions", nargs="+",
                    help="session dir(s), or root dir(s) containing sessions")
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--results", default="all",
                    help="comma list of labels to export: all|success|fail|unlabeled")
    ap.add_argument("--teleop-only", action="store_true",
                    help="keep only frames inside arbiter_mode==1 (teleop) spans")
    ap.add_argument("--camera", default="head", help="video source camera name")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--chunks-size", type=int, default=1000)
    ap.add_argument("--task-category", default="KIST")
    args = ap.parse_args()

    wanted = {r.strip() for r in args.results.split(",")}
    if "all" in wanted:
        wanted = {"success", "fail", "unlabeled"}

    out = Path(args.out).expanduser()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"error: output dir not empty: {out} (refusing to mix datasets)")

    episodes, tasks, task_index_of = [], [], {}
    ep_index = global_index = 0

    for session in discover_sessions(args.sessions):
        meta = parse_meta(session)
        label = meta["result"] or "unlabeled"
        if label not in wanted:
            print(f"[skip] {session.name}: result={label}")
            continue
        cam = session / args.camera
        if not (cam / "color.idx.csv").is_file():
            print(f"[skip] {session.name}: no {args.camera} camera stream")
            continue

        decoded = decode_seqs(cam)
        df, kept = build_rows(session, args.camera, decoded, args.teleop_only)
        if df.empty:
            print(f"[skip] {session.name}: 0 usable frames"
                  + (" (no teleop span?)" if args.teleop_only else ""))
            continue

        if meta["task"] not in task_index_of:
            task_index_of[meta["task"]] = len(tasks)
            tasks.append({"task_index": len(tasks), "task": meta["task"],
                          "category": args.task_category,
                          "description": meta["task"]})
        t_idx = task_index_of[meta["task"]]

        chunk = ep_index // args.chunks_size
        stats = make_parquet(
            df, ep_index, t_idx, global_index,
            out / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_index:06d}.parquet",
            args.fps)
        n_vid = write_video(
            cam, set(kept),
            out / "videos" / f"chunk-{chunk:03d}" / "egocentric"
                / f"episode_{ep_index:06d}.mp4",
            args.fps)
        if n_vid != len(df):
            sys.exit(f"error: {session.name}: video frames ({n_vid}) != "
                     f"parquet rows ({len(df)})")

        episodes.append({
            "episode_index": ep_index, "tasks": [t_idx], "length": len(df),
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + len(df) - 1,
            "robot_type": "g1", "instruction": meta["task"],
            "result": label, "session": session.name, "stats": stats,
        })
        print(f"[ok]   {session.name}: {len(df)} frames, result={label}, "
              f'task="{meta["task"]}"')
        global_index += len(df)
        ep_index += 1

    if not episodes:
        sys.exit("error: no episodes exported")
    write_meta(out, episodes, tasks, args.fps, args.chunks_size)
    print(f"\nwrote {len(episodes)} episode(s), "
          f"{sum(e['length'] for e in episodes)} frames -> {out}")


if __name__ == "__main__":
    main()
