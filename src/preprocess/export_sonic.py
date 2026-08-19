#!/usr/bin/env python3
"""kist session(s) -> GR00T N1.7 fine-tuning dataset, UNITREE_G1_SONIC form.

Target: the GR00T-WholeBodyControl "Sonic VLA" data format
(nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html),
consumed by  launch_finetune.py --embodiment-tag UNITREE_G1_SONIC.
Feature names/dtypes/joint order follow the repo's own
gear_sonic/data/features_sonic_vla.py; the embodiment's modality contract
(Isaac-GR00T embodiment_configs.py "unitree_g1_sonic") consumes:

  state : 7 joint groups of observation.state[43] + projected_gravity
  action: action.motion_token[64] + teleop.{left,right}_hand_joints[7]
  video : ego_view          language: annotation.human.task_description

Timestep grid = one motion_token tick (50 Hz, the SONIC decoder cadence,
arbiter_mode==1 rows only). The VLA learns 40-step token chunks — spacing
them at the decoder's own 50 Hz is what keeps deployment cadence correct,
so the token stream is the master clock here, NOT the camera:

  motion_token 50 Hz  -> row (exact, this IS the action)
  lowstate/hands 1k/830 Hz -> nearest recv_ns   (state)
  lowcmd/hand_cmd     -> backward recv_ns       (standing command; action.wbc)
  head camera 30 fps  -> nearest frame, written repeated into a 50 fps mp4
                         (frame count == row count stays exact)

observation.state / action.wbc are in gear_sonic's URDF traversal order:
legs(12) waist(3) left_arm(7) LEFT_HAND(7) right_arm(7) RIGHT_HAND(7);
within a hand URDF order is thumb0,1,2, middle0,1, index0,1 while the Dex3
SDK wire order is thumb0,1,2, index0,1, middle0,1 (per gear_sonic's
supplemental_info) — the scatter below reorders. teleop.*_hand_joints stay
in raw SDK order (that is what the hand cmd topic consumes at deployment).

Usage:
  python export_sonic.py SESSIONS_ROOT --out DATASET_DIR [--results success]
      [--camera head] [--fps 50] [--min-rows 25]
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
import export_lerobot as base  # session discovery, H.264 decode, asof join

BODY = [f"m{i:02d}" for i in range(29)]     # lowstate/lowcmd motor order
HAND = [f"f{i}" for i in range(7)]          # Dex3 SDK wire order
TOKEN_DIM = 64
TOKENS = [f"t{i:02d}" for i in range(TOKEN_DIM)]

# SDK hand motor i lands at URDF slice position SCATTER[i]
# (SDK: thumb0,1,2, index0,1, middle0,1 / URDF: thumb0,1,2, middle0,1, index0,1)
HAND_SDK_TO_URDF = [0, 1, 2, 5, 6, 3, 4]

_BODY_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
_HAND_URDF = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1",
              "index_0", "index_1"]
JOINT_NAMES = (_BODY_NAMES[:22]
               + [f"left_hand_{n}_joint" for n in _HAND_URDF]
               + _BODY_NAMES[22:]
               + [f"right_hand_{n}_joint" for n in _HAND_URDF])  # 43, URDF order

# observation.state / action.wbc slice map (URDF order)
GROUPS = {"left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15),
          "left_arm": (15, 22), "left_hand": (22, 29),
          "right_arm": (29, 36), "right_hand": (36, 43)}

VIDEO_KEY = "observation.images.ego_view"
# --wrist-cameras: optional extra views, keys per the WBC repo's own
# get_wrist_camera_features() (the pre-registered unitree_g1_sonic embodiment
# consumes ego_view only — these ride along for custom embodiment configs)
WRIST_KEYS = {"left_wrist": "observation.images.left_wrist",
              "right_wrist": "observation.images.right_wrist"}


def assemble_43(body: np.ndarray, lh: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """(n,29)+(n,7)+(n,7) SDK-order readings -> (n,43) URDF-order vector."""
    n = body.shape[0]
    out = np.empty((n, 43), dtype=np.float64)
    out[:, 0:22] = body[:, 0:22]          # legs+waist+left arm
    out[:, 29:36] = body[:, 22:29]        # right arm
    for i, p in enumerate(HAND_SDK_TO_URDF):
        out[:, 22 + p] = lh[:, i]
        out[:, 36 + p] = rh[:, i]
    return out


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Gravity [0,0,-1] rotated into the body frame (quat inverse rotate)."""
    w, x, y, z = (quat_wxyz[:, i] for i in range(4))
    # R(q)^T @ [0,0,-1] — third row of R(q), negated
    gx = -(2 * (x * z - w * y))
    gy = -(2 * (y * z + w * x))
    gz = -(1 - 2 * (x * x + y * y))
    return np.stack([gx, gy, gz], axis=1)


def build_rows(session: Path, camera: str, decoded_seqs: list):
    """Rows = motion_token ticks (arbiter_mode==1). Joins by recv_ns."""
    tok = pd.read_csv(session / "motion_token.csv")
    tok = tok[tok["arbiter_mode"] == 1].astype({"recv_ns": "int64"})
    if tok.empty:
        return tok, None
    gaps = int((tok["seq"].diff().dropna() > 1).sum())
    if gaps:
        print(f"  경고: motion_token seq 갭 {gaps}곳 (CONTROL 이탈 구간 포함)")
    df = tok[["recv_ns"] + TOKENS].sort_values("recv_ns").reset_index(drop=True)

    quat = ["quat_w", "quat_x", "quat_y", "quat_z"]
    specs = [
        (session / "lowstate.csv",
         [f"{m}_q" for m in BODY] + quat, "st", "nearest"),
        (session / "hand_left.csv", [f"{f}_q" for f in HAND], "hl", "nearest"),
        (session / "hand_right.csv", [f"{f}_q" for f in HAND], "hr", "nearest"),
        (session / "lowcmd.csv", [f"{m}_q" for m in BODY], "ca", "backward"),
        (session / "hand_cmd_left.csv", [f"{f}_q" for f in HAND], "cl", "backward"),
        (session / "hand_cmd_right.csv", [f"{f}_q" for f in HAND], "cr", "backward"),
    ]
    for csv, cols, tag, direction in specs:
        df = base._asof(df, csv, cols, direction).rename(
            columns={c: f"{tag}.{c}" for c in cols})

    # nearest decodable head frame per row (video timeline)
    idx = pd.read_csv(session / camera / "color.idx.csv",
                      usecols=["seq", "recv_ns"]).astype("int64")
    frames = idx[idx["seq"].isin(decoded_seqs)].sort_values("recv_ns")
    df = pd.merge_asof(df, frames.rename(columns={"seq": "frame_seq",
                                                  "recv_ns": "frame_ns"}),
                       left_on="recv_ns", right_on="frame_ns",
                       direction="nearest")
    df = df.dropna().reset_index(drop=True)
    return df, df["frame_seq"].astype(int).to_numpy()


def nearest_frame_seqs(camera_dir: Path, decoded_seqs: list,
                       row_ns: np.ndarray) -> np.ndarray:
    """Per-row nearest decodable frame of an extra camera (own seq space)."""
    idx = pd.read_csv(camera_dir / "color.idx.csv",
                      usecols=["seq", "recv_ns"]).astype("int64")
    frames = idx[idx["seq"].isin(decoded_seqs)].sort_values("recv_ns")
    rows = pd.DataFrame({"recv_ns": row_ns.astype("int64")})
    m = pd.merge_asof(rows, frames, on="recv_ns", direction="nearest")
    return m["seq"].astype(int).to_numpy()


def write_video(camera_dir: Path, frame_seqs: np.ndarray, out_path: Path,
                fps: int):
    """One mp4 frame per row: 30 fps camera frames repeated onto the token
    grid. nearest-in-time assignment is monotonic, so a single streaming
    decode pass suffices: each decoded frame is written `count(seq)` times."""
    counts = pd.Series(frame_seqs).value_counts().to_dict()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc, written, samples = None, 0, []
    for seq, frame in base._decode(camera_dir):
        n = counts.get(seq, 0)
        if n == 0:
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
        proc.stdin.write(img.tobytes() * n)
        if (written // 10) != ((written + n) // 10):
            samples.append(img[:, :, ::-1].astype(np.float32) / 255.0)
        written += n
    if proc is not None:
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit(f"error: ffmpeg failed for {out_path}")
    px = np.concatenate([s.reshape(-1, 3) for s in samples])
    s311 = lambda v: v.reshape(3, 1, 1).tolist()
    stats = {"min": s311(px.min(0)), "max": s311(px.max(0)),
             "mean": s311(px.mean(0)), "std": s311(px.std(0)),
             "count": [len(samples)]}
    return written, stats


def assemble_features(df: pd.DataFrame) -> dict:
    grab = lambda cols, dt: df[cols].to_numpy(dtype=dt)
    hand = lambda tag: [f"{tag}.{f}_q" for f in HAND]
    quat = grab([f"st.quat_{c}" for c in "wxyz"], np.float64)
    return {
        "observation.state": assemble_43(
            grab([f"st.{m}_q" for m in BODY], np.float64),
            grab(hand("hl"), np.float64), grab(hand("hr"), np.float64)),
        "action.wbc": assemble_43(
            grab([f"ca.{m}_q" for m in BODY], np.float64),
            grab(hand("cl"), np.float64), grab(hand("cr"), np.float64)),
        "observation.root_orientation": quat,
        "observation.projected_gravity": projected_gravity(quat),
        "observation.init_base_quat": np.tile(quat[0], (len(df), 1)),
        "action.motion_token": grab(TOKENS, np.float64),
        "teleop.left_hand_joints": grab(hand("cl"), np.float32),
        "teleop.right_hand_joints": grab(hand("cr"), np.float32),
    }


FLOAT32_KEYS = {"teleop.left_hand_joints", "teleop.right_hand_joints"}


def write_parquet(vecs: dict, n: int, ep: int, task: int, start: int,
                  fps: int, path: Path) -> dict:
    stats, cols = {}, {}
    for k, v in vecs.items():
        stats[k] = {"min": v.min(0).tolist(), "max": v.max(0).tolist(),
                    "mean": v.mean(0).tolist(), "std": v.std(0).tolist(),
                    "count": [n]}
        typ = pa.float32() if k in FLOAT32_KEYS else pa.float64()
        cols[k] = pa.array(list(v.astype(
            np.float32 if k in FLOAT32_KEYS else np.float64)),
            type=pa.list_(typ))
    scalars = {
        "timestamp": (np.arange(n) / fps).astype(np.float32),
        "frame_index": np.arange(n), "episode_index": np.full(n, ep),
        "index": np.arange(start, start + n), "task_index": np.full(n, task),
    }
    for k, v in scalars.items():
        cols[k] = pa.array(v, type=pa.float32() if v.dtype == np.float32
                           else pa.int64())
        stats[k] = {"min": [float(v.min())], "max": [float(v.max())],
                    "mean": [float(v.mean())], "std": [float(v.std())],
                    "count": [n]}
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({k: cols[k] for k in
                             list(vecs) + list(scalars)}), path)
    return stats


def write_meta(out: Path, episodes: list, tasks: list, fps: int,
               chunks_size: int, wrist_cams: list):
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    n_ep = len(episodes)

    features = {
        VIDEO_KEY: {"dtype": "video", "shape": [480, 640, 3],
                    "names": ["height", "width", "channel"],
                    "info": {"video.height": 480, "video.width": 640,
                             "video.codec": "h264",
                             "video.pix_fmt": "yuv420p",
                             "video.is_depth_map": False, "video.fps": fps,
                             "video.channels": 3, "has_audio": False}},
        "observation.state": {"dtype": "float64", "shape": [43],
                              "names": JOINT_NAMES},
        "action.wbc": {"dtype": "float64", "shape": [43],
                       "names": JOINT_NAMES},
        "observation.root_orientation": {
            "dtype": "float64", "shape": [4],
            "names": ["base_qw", "base_qx", "base_qy", "base_qz"]},
        "observation.projected_gravity": {
            "dtype": "float64", "shape": [3],
            "names": ["gravity_x", "gravity_y", "gravity_z"]},
        "observation.init_base_quat": {
            "dtype": "float64", "shape": [4],
            "names": ["init_base_qw", "init_base_qx", "init_base_qy",
                      "init_base_qz"]},
        "action.motion_token": {"dtype": "float64", "shape": [TOKEN_DIM],
                                "names": "motion_token"},
        "teleop.left_hand_joints": {"dtype": "float32", "shape": [7],
                                    "names": "left_hand_joints"},
        "teleop.right_hand_joints": {"dtype": "float32", "shape": [7],
                                     "names": "right_hand_joints"},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for cam in wrist_cams:
        features[WRIST_KEYS[cam]] = json.loads(
            json.dumps(features[VIDEO_KEY]))  # same shape/codec info
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "robot_type": "unitree_g1",
        "total_episodes": n_ep,
        "total_frames": sum(e["length"] for e in episodes),
        "total_tasks": len(tasks), "total_videos": n_ep,
        "total_chunks": math.ceil(n_ep / chunks_size),
        "chunks_size": chunks_size, "fps": fps,
        "splits": {"train": f"0:{n_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features}, indent=4))

    sl = lambda a, b, **kw: {"start": a, "end": b, **kw}
    modality = {
        "state": {
            **{g: sl(a, b) for g, (a, b) in GROUPS.items()},
            "root_orientation": sl(0, 4,
                original_key="observation.root_orientation",
                rotation_type="quaternion"),
            "projected_gravity": sl(0, 3,
                original_key="observation.projected_gravity"),
            "init_base_quat": sl(0, 4,
                original_key="observation.init_base_quat",
                rotation_type="quaternion"),
        },
        "action": {
            "motion_token": sl(0, TOKEN_DIM,
                               original_key="action.motion_token"),
            "left_hand_joints": sl(0, 7,
                                   original_key="teleop.left_hand_joints"),
            "right_hand_joints": sl(0, 7,
                                    original_key="teleop.right_hand_joints"),
        },
        "video": {"ego_view": {"original_key": VIDEO_KEY},
                  **{cam: {"original_key": WRIST_KEYS[cam]}
                     for cam in wrist_cams}},
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
        description="kist session(s) -> UNITREE_G1_SONIC dataset "
                    "(motion_token 50 Hz grid)")
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default="all",
                    help="all|success|fail|unlabeled (comma list)")
    ap.add_argument("--camera", default="head")
    ap.add_argument("--fps", type=int, default=50,
                    help="token tick rate (gearsonic publishes 50 Hz)")
    ap.add_argument("--chunks-size", type=int, default=1000)
    ap.add_argument("--min-rows", type=int, default=25,
                    help="skip episodes with fewer token rows")
    ap.add_argument("--wrist-cameras", action="store_true",
                    help="also export left_wrist/right_wrist videos")
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
        df, frame_seqs = build_rows(session, args.camera, decoded)
        if frame_seqs is None or len(df) < args.min_rows:
            print(f"[skip] {session.name}: token rows {len(df)} < "
                  f"{args.min_rows}")
            continue

        if meta["task"] not in task_index_of:
            task_index_of[meta["task"]] = len(tasks)
            tasks.append({"task_index": len(tasks), "task": meta["task"]})
        t = task_index_of[meta["task"]]
        chunk = ep // args.chunks_size

        vecs = assemble_features(df)
        stats = write_parquet(
            vecs, len(df), ep, t, start, args.fps,
            out / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet")
        n_vid, vstats = write_video(
            cam, frame_seqs,
            out / "videos" / f"chunk-{chunk:03d}" / VIDEO_KEY
                / f"episode_{ep:06d}.mp4", args.fps)
        if n_vid != len(df):
            sys.exit(f"error: {session.name}: video frames ({n_vid}) != "
                     f"rows ({len(df)})")
        stats[VIDEO_KEY] = vstats

        if args.wrist_cameras:
            for cam_name, key in WRIST_KEYS.items():
                wdir = session / cam_name
                if not (wdir / "color.idx.csv").is_file():
                    print(f"  경고: {cam_name} 스트림 없음 — 생략")
                    continue
                w_decoded = base.decode_seqs(wdir)
                w_seqs = nearest_frame_seqs(
                    wdir, w_decoded, df["recv_ns"].to_numpy())
                n_w, w_stats = write_video(
                    wdir, w_seqs,
                    out / "videos" / f"chunk-{chunk:03d}" / key
                        / f"episode_{ep:06d}.mp4", args.fps)
                if n_w != len(df):
                    sys.exit(f"error: {session.name}/{cam_name}: video "
                             f"frames ({n_w}) != rows ({len(df)})")
                stats[key] = w_stats

        episodes.append({"episode_index": ep, "tasks": [meta["task"]],
                         "length": len(df), "result": label,
                         "session": session.name, "stats": stats})
        print(f"[ok]   {session.name}: {len(df)} token rows @ {args.fps} Hz, "
              f"result={label}")
        start += len(df)
        ep += 1

    if not episodes:
        sys.exit("error: no episodes exported")
    write_meta(out, episodes, tasks, args.fps, args.chunks_size,
               list(WRIST_KEYS) if args.wrist_cameras else [])
    print(f"\nwrote {len(episodes)} episode(s), "
          f"{sum(e['length'] for e in episodes)} rows -> {out}")


if __name__ == "__main__":
    main()
