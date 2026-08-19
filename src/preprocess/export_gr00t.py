#!/usr/bin/env python3
"""kist session(s) -> GR00T N1.7 fine-tuning dataset (AppleToPlate schema).

Produces the exact on-disk form of nvidia/GR00T-N1.7-AppleToPlate — the
dataset the official G1 end-to-end tutorial feeds to launch_finetune.py —
so the output plugs into that command with only the path changed. Every
schema detail below (feature names/dims/dtypes, meta layout, modality
groups, constants) was copied from that dataset's actual files, not from
the prose docs.

  observation.state [43] = lowstate m00..m28 q  (legs 12, waist 3, arms 14)
                         + hand_left f0..f6 q + hand_right f0..f6 q
  action            [43] = lowcmd   m00..m28 q + hand_cmd L q + hand_cmd R q
  action.effort_<group>  = the same sources' tau (feed-forward torque)
  action.navigate_command    [3] = zeros    (A2P: all zeros — stationary task)
  action.base_height_command [1] = 0.72 m   (A2P: constant; --base-height)
  observation.images.ego_view    = head camera mp4 (h264 here; A2P used AV1 —
                                   loaders decode either, info.json records ours)

Alignment front-end (frame grid, recv_ns joins, filters) is shared with
export_lerobot.py: head frame = timestep, states nearest, commands backward.
--results / --teleop-only behave identically; result/session labels are
kept as extra keys in meta/episodes.jsonl.

Usage:
  python export_gr00t.py SESSIONS_ROOT --out DATASET_DIR [--results success]
      [--teleop-only] [--camera head] [--fps 30] [--base-height 0.72]
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_lerobot as base  # session discovery, decode, asof join

BODY = [f"m{i:02d}" for i in range(29)]      # lowstate/lowcmd motors 0..28
HAND = [f"f{i}" for i in range(7)]

# body-motor slices of the 29-dim body vectors, in A2P group order
BODY_GROUPS = {"left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15),
               "left_arm": (15, 22), "right_arm": (22, 29)}

VIDEO_KEY = "observation.images.ego_view"
EFFORT_KEYS = [f"action.effort_{g}" for g in
               ["left_leg", "right_leg", "waist", "left_arm", "right_arm",
                "left_hand", "right_hand"]]
VEC_KEYS = (["observation.state", "action"] + EFFORT_KEYS
            + ["action.navigate_command", "action.base_height_command"])


def build_rows(session: Path, camera: str, decoded: list, teleop_only: bool):
    """Same join recipe as export_lerobot.build_rows, but pulling the full
    body (q for state, q+tau for commands) instead of arm-only columns."""
    idx = pd.read_csv(session / camera / "color.idx.csv",
                      usecols=["seq", "recv_ns"]).astype("int64")
    df = idx[idx["seq"].isin(decoded)].sort_values("recv_ns").reset_index(drop=True)

    specs = [  # (csv, columns, tag, direction)
        (session / "lowstate.csv", [f"{m}_q" for m in BODY], "st", "nearest"),
        (session / "hand_left.csv", [f"{f}_q" for f in HAND], "hl", "nearest"),
        (session / "hand_right.csv", [f"{f}_q" for f in HAND], "hr", "nearest"),
        (session / "lowcmd.csv",
         [f"{m}_{f}" for m in BODY for f in ("q", "tau")], "ca", "backward"),
        (session / "hand_cmd_left.csv",
         [f"{f}_{k}" for f in HAND for k in ("q", "tau")], "cl", "backward"),
        (session / "hand_cmd_right.csv",
         [f"{f}_{k}" for f in HAND for k in ("q", "tau")], "cr", "backward"),
    ]
    for csv, cols, tag, direction in specs:
        df = base._asof(df, csv, cols, direction).rename(
            columns={c: f"{tag}.{c}" for c in cols})

    if teleop_only:
        tok = pd.read_csv(session / "motion_token.csv",
                          usecols=["recv_ns", "arbiter_mode"])
        tok = tok[tok["arbiter_mode"] == 1].astype({"recv_ns": "int64"})
        if tok.empty:
            return df.iloc[0:0], []
        near = pd.merge_asof(df[["recv_ns"]], tok.sort_values("recv_ns"),
                             on="recv_ns", direction="nearest",
                             tolerance=40_000_000)
        df = df[near["arbiter_mode"].notna().values]

    df = df.dropna()
    return df.reset_index(drop=True), df["seq"].astype(int).tolist()


def assemble(df: pd.DataFrame, base_height: float) -> dict:
    """A2P feature vectors (float32 matrices, one row per timestep)."""
    n = len(df)
    grab = lambda cols: df[cols].to_numpy(dtype=np.float32)
    hands = lambda tag, f: [f"{tag}.{j}_{f}" for j in HAND]

    out = {
        "observation.state": np.hstack([
            grab([f"st.{m}_q" for m in BODY]),
            grab(hands("hl", "q")), grab(hands("hr", "q"))]),
        "action": np.hstack([
            grab([f"ca.{m}_q" for m in BODY]),
            grab(hands("cl", "q")), grab(hands("cr", "q"))]),
        "action.navigate_command": np.zeros((n, 3), dtype=np.float32),
        "action.base_height_command": np.full((n, 1), base_height, np.float32),
    }
    for g, (a, b) in BODY_GROUPS.items():
        out[f"action.effort_{g}"] = grab([f"ca.{m}_tau" for m in BODY[a:b]])
    out["action.effort_left_hand"] = grab(hands("cl", "tau"))
    out["action.effort_right_hand"] = grab(hands("cr", "tau"))
    return out


def write_video_with_stats(camera_dir: Path, keep: set, out_path: Path,
                           fps: int):
    """base.write_video + per-channel pixel stats (episodes_stats needs an
    entry for the video feature too — A2P has one). Stats over every 10th
    kept frame, normalized to [0,1], shaped (3,1,1) like lerobot's."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc, written, samples = None, 0, []
    for seq, frame in base._decode(camera_dir):
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
        if written % 10 == 0:
            samples.append(img[:, :, ::-1].astype(np.float32) / 255.0)  # RGB
        written += 1
    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit(f"error: ffmpeg failed for {out_path}")
    px = np.concatenate([s.reshape(-1, 3) for s in samples])
    shape311 = lambda v: v.reshape(3, 1, 1).tolist()
    stats = {"min": shape311(px.min(0)), "max": shape311(px.max(0)),
             "mean": shape311(px.mean(0)), "std": shape311(px.std(0)),
             "count": [len(samples)]}
    return written, stats


def feature_stats(mat: np.ndarray, n: int) -> dict:
    return {"min": mat.min(0).tolist(), "max": mat.max(0).tolist(),
            "mean": mat.mean(0).tolist(), "std": mat.std(0).tolist(),
            "count": [n]}


def write_parquet(vecs: dict, n: int, ep: int, task: int, start: int,
                  fps: int, path: Path) -> dict:
    """A2P column set/order; returns the episode's per-feature stats."""
    stats = {k: feature_stats(v, n) for k, v in vecs.items()}
    cols = {k: pa.array(list(v), type=pa.list_(pa.float32()))
            for k, v in vecs.items()}
    scalars = {
        "timestamp": (np.arange(n) / fps).astype(np.float32),
        "frame_index": np.arange(n), "episode_index": np.full(n, ep),
        "index": np.arange(start, start + n), "task_index": np.full(n, task),
    }
    for k, v in scalars.items():
        typ = pa.float32() if v.dtype == np.float32 else pa.int64()
        cols[k] = pa.array(v, type=typ)
        stats[k] = {"min": [float(v.min())], "max": [float(v.max())],
                    "mean": [float(v.mean())], "std": [float(v.std())],
                    "count": [n]}
    order = VEC_KEYS + list(scalars)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: cols[k] for k in order}), path)
    return stats


def write_meta(out: Path, episodes: list, tasks: list, fps: int,
               chunks_size: int):
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    n_ep, n_fr = len(episodes), sum(e["length"] for e in episodes)

    dims = {"observation.state": 43, "action": 43,
            "action.effort_left_leg": 6, "action.effort_right_leg": 6,
            "action.effort_waist": 3, "action.effort_left_arm": 7,
            "action.effort_right_arm": 7, "action.effort_left_hand": 7,
            "action.effort_right_hand": 7, "action.navigate_command": 3,
            "action.base_height_command": 1}
    features = {k: {"dtype": "float32", "shape": [d],
                    "names": ["observation_to_policy" if k.startswith("obser")
                              else "action_from_policy"]}
                for k, d in dims.items()}
    features[VIDEO_KEY] = {
        "dtype": "video", "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "info": {"video.height": 480, "video.width": 640,
                 "video.codec": "h264", "video.pix_fmt": "yuv420p",
                 "video.is_depth_map": False, "video.fps": fps,
                 "video.channels": 3, "has_audio": False}}
    for k in ("timestamp", "frame_index", "episode_index", "index",
              "task_index"):
        features[k] = {"dtype": "float32" if k == "timestamp" else "int64",
                       "shape": [1], "names": None}

    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "robot_type": "unitree_g1",
        "total_episodes": n_ep, "total_frames": n_fr,
        "total_tasks": len(tasks), "total_videos": n_ep,
        "total_chunks": math.ceil(n_ep / chunks_size),
        "chunks_size": chunks_size, "fps": fps,
        "splits": {"train": f"0:{n_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features}, indent=4))

    sl = lambda a, b: {"start": a, "end": b}
    groups = {**{g: sl(a, b) for g, (a, b) in BODY_GROUPS.items()},
              "left_hand": sl(29, 36), "right_hand": sl(36, 43)}
    modality = {
        "state": groups,
        "action": {**groups,
                   **{k.split("action.")[1]:
                      {"original_key": k, **sl(0, dims[k])}
                      for k in EFFORT_KEYS
                      + ["action.navigate_command",
                         "action.base_height_command"]}},
        "video": {"ego_view": {"original_key": VIDEO_KEY}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=4))

    with open(meta / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    with open(meta / "episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps({k: v for k, v in e.items() if k != "stats"})
                    + "\n")
    with open(meta / "episodes_stats.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps({"episode_index": e["episode_index"],
                                "stats": e["stats"]}) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="kist session(s) -> GR00T N1.7 dataset (AppleToPlate schema)")
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default="all",
                    help="all|success|fail|unlabeled (comma list)")
    ap.add_argument("--teleop-only", action="store_true")
    ap.add_argument("--camera", default="head")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--chunks-size", type=int, default=1000)
    ap.add_argument("--base-height", type=float, default=0.72,
                    help="constant base_height_command (m), A2P uses 0.72")
    args = ap.parse_args()

    wanted = {r.strip() for r in args.results.split(",")}
    if "all" in wanted:
        wanted = {"success", "fail", "unlabeled"}
    out = Path(args.out).expanduser()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"error: output dir not empty: {out}")

    episodes, tasks, task_index_of = [], [], {}
    ep = start = 0
    for session in base.discover_sessions(args.sessions):
        meta = base.parse_meta(session)
        label = meta["result"] or "unlabeled"
        if label not in wanted:
            print(f"[skip] {session.name}: result={label}")
            continue
        cam = session / args.camera
        if not (cam / "color.idx.csv").is_file():
            print(f"[skip] {session.name}: no {args.camera} camera stream")
            continue

        decoded = base.decode_seqs(cam)
        df, kept = build_rows(session, args.camera, decoded, args.teleop_only)
        if df.empty:
            print(f"[skip] {session.name}: 0 usable frames")
            continue

        if meta["task"] not in task_index_of:
            task_index_of[meta["task"]] = len(tasks)
            tasks.append({"task_index": len(tasks), "task": meta["task"]})
        t = task_index_of[meta["task"]]
        chunk = ep // args.chunks_size

        vecs = assemble(df, args.base_height)
        stats = write_parquet(
            vecs, len(df), ep, t, start, args.fps,
            out / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet")
        n_vid, vstats = write_video_with_stats(
            cam, set(kept),
            out / "videos" / f"chunk-{chunk:03d}" / VIDEO_KEY
                / f"episode_{ep:06d}.mp4", args.fps)
        if n_vid != len(df):
            sys.exit(f"error: {session.name}: video frames ({n_vid}) != "
                     f"rows ({len(df)})")
        stats[VIDEO_KEY] = vstats

        episodes.append({"episode_index": ep, "tasks": [meta["task"]],
                         "length": len(df), "result": label,
                         "session": session.name, "stats": stats})
        print(f"[ok]   {session.name}: {len(df)} frames, result={label}")
        start += len(df)
        ep += 1

    if not episodes:
        sys.exit("error: no episodes exported")
    write_meta(out, episodes, tasks, args.fps, args.chunks_size)
    print(f"\nwrote {len(episodes)} episode(s), "
          f"{sum(e['length'] for e in episodes)} frames -> {out}")


if __name__ == "__main__":
    main()
