#!/usr/bin/env python3
"""export_sonic.py + leading-idle trimming. Same output format, same CLI,
plus trim options — export_sonic.py itself is untouched (helpers imported).

Per episode, ONLY the still stretch at the START (before the right arm first
moves) is dropped, minus --pad seconds kept as pre-motion context. Nothing
after the first movement is touched — trailing stillness stays in.

Motion-start detection is DISPLACEMENT-based, not velocity-based: teleop
tracking jitter keeps the arm "moving" a little the whole time, so speed
thresholds misfire. Instead, the start is the first row where the MEASURED
right arm (lowstate m22..m28) has moved more than --arm-disp rad away from
its row-0 pose, OR the right hand (Dex3) more than --hand-disp rad —
whichever comes first (in this data the fist often closes before the reach).

Usage:
  python export_sonic_trim.py SESSIONS_ROOT --out DATASET_DIR
      [--arm-disp 0.15] [--hand-disp 0.1] [--pad 0.5]
      [--no-trim] [--dry-run] [+ all export_sonic.py options]

--no-trim disables the leading-idle cut entirely (every session keeps all
rows, nothing is skipped for "never moved") while keeping the rest of this
exporter (camera-fps filter, min-rows, episodes.jsonl fields) identical.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_lerobot as base
import export_sonic as sonic

RIGHT_ARM_COLS = [f"st.m{i}_q" for i in range(22, 29)]   # URDF right_arm
RIGHT_HAND_COLS = [f"hr.{f}_q" for f in sonic.HAND]


def camera_fps(cam_dir: Path) -> float:
    """Actual head-camera receive rate from color.idx.csv (Hz)."""
    idx = pd.read_csv(cam_dir / "color.idx.csv", usecols=["recv_ns"])
    ts = idx["recv_ns"].to_numpy()
    if len(ts) < 2:
        return 0.0
    return (len(ts) - 1) / ((ts[-1] - ts[0]) / 1e9)


def motion_start(df: pd.DataFrame, fps: int, args) -> int | None:
    """First row where arm or hand has left its starting pose (None = never)."""
    arm = df[RIGHT_ARM_COLS].to_numpy(dtype=np.float64)
    hand = df[RIGHT_HAND_COLS].to_numpy(dtype=np.float64)
    moved = (np.abs(arm - arm[0]).max(axis=1) > args.arm_disp) \
        | (np.abs(hand - hand[0]).max(axis=1) > args.hand_disp)
    idx = np.flatnonzero(moved)
    return int(idx[0]) if idx.size else None


def main():
    ap = argparse.ArgumentParser(
        description="export_sonic + leading-idle trimming")
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default="all")
    ap.add_argument("--camera", default="head")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--chunks-size", type=int, default=1000)
    ap.add_argument("--min-rows", type=int, default=50,
                    help="drop trimmed episodes shorter than this "
                         "(50 = 1 s; the VLA chunk needs 40)")
    ap.add_argument("--wrist-cameras", action="store_true")
    # trimming knobs
    ap.add_argument("--arm-disp", type=float, default=0.15,
                    help="right-arm displacement from the start pose that "
                         "marks motion start (rad)")
    ap.add_argument("--hand-disp", type=float, default=0.1,
                    help="right-hand displacement that marks motion start "
                         "(rad; the fist often closes before the reach)")
    ap.add_argument("--pad", type=float, default=0.5,
                    help="still context kept before the motion start (s)")
    ap.add_argument("--min-cam-fps", type=float, default=25.0,
                    help="skip sessions whose head camera received fewer "
                         "than this many frames/s (degraded stream; 0=off)")
    ap.add_argument("--no-trim", action="store_true",
                    help="트리밍 비활성화: 앞부분을 잘라내지 않고 전체 행을 "
                         "그대로 내보냄 (동작 미검출 스킵도 하지 않음)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report per-session trim, write nothing")
    args = ap.parse_args()

    wanted = {r.strip() for r in args.results.split(",")}
    if "all" in wanted:
        wanted = {"success", "fail", "unlabeled"}
    out = Path(args.out).expanduser()
    if not args.dry_run and out.exists() and any(out.iterdir()):
        sys.exit(f"error: output dir not empty: {out}")

    episodes, tasks, task_index_of = [], [], {}
    ep = start = 0
    rows_in = rows_kept = 0
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

        if args.min_cam_fps > 0:
            cfps = camera_fps(cam)
            if cfps < args.min_cam_fps:
                print(f"[skip] {session.name}: {args.camera} 카메라 "
                      f"{cfps:.1f}fps < {args.min_cam_fps:g}fps (저하 스트림)")
                continue

        decoded = base.decode_seqs(cam)
        df, frame_seqs = sonic.build_rows(session, args.camera, decoded)
        if frame_seqs is None or len(df) == 0:
            print(f"[skip] {session.name}: no token rows")
            continue

        if args.no_trim:
            cut = 0
            print(f"[keep] {session.name}: 트리밍 없음, {len(df)}행 유지")
        else:
            first = motion_start(df, args.fps, args)
            if first is None:
                print(f"[skip] {session.name}: 오른팔/오른손이 시작 자세를 벗어난 적 없음")
                continue
            cut = max(0, first - int(round(args.pad * args.fps)))
            print(f"[trim] {session.name}: 앞 {cut}행({cut / args.fps:.1f}s) 제거, "
                  f"{len(df)} -> {len(df) - cut}행")
        rows_in += len(df)
        rows_kept += len(df) - cut
        if len(df) - cut < args.min_rows:
            print(f"[skip] {session.name}: 트리밍 후 {len(df) - cut}행 < "
                  f"{args.min_rows}")
            continue
        if args.dry_run:
            continue

        sdf = df.iloc[cut:].reset_index(drop=True)
        sseqs = frame_seqs[cut:]

        if meta["task"] not in task_index_of:
            task_index_of[meta["task"]] = len(tasks)
            tasks.append({"task_index": len(tasks), "task": meta["task"]})
        t = task_index_of[meta["task"]]
        chunk = ep // args.chunks_size

        vecs = sonic.assemble_features(sdf)
        stats = sonic.write_parquet(
            vecs, len(sdf), ep, t, start, args.fps,
            out / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet")
        n_vid, vstats = sonic.write_video(
            cam, sseqs,
            out / "videos" / f"chunk-{chunk:03d}" / sonic.VIDEO_KEY
                / f"episode_{ep:06d}.mp4", args.fps)
        if n_vid != len(sdf):
            sys.exit(f"error: {session.name}: video frames ({n_vid}) != "
                     f"rows ({len(sdf)})")
        stats[sonic.VIDEO_KEY] = vstats

        if args.wrist_cameras:
            for cam_name, key in sonic.WRIST_KEYS.items():
                wdir = session / cam_name
                if not (wdir / "color.idx.csv").is_file():
                    print(f"  경고: {cam_name} 스트림 없음 — 생략")
                    continue
                w_decoded = base.decode_seqs(wdir)
                w_seqs = sonic.nearest_frame_seqs(
                    wdir, w_decoded, sdf["recv_ns"].to_numpy())
                n_w, w_stats = sonic.write_video(
                    wdir, w_seqs,
                    out / "videos" / f"chunk-{chunk:03d}" / key
                        / f"episode_{ep:06d}.mp4", args.fps)
                if n_w != len(sdf):
                    sys.exit(f"error: {session.name}/{cam_name}: video "
                             f"frames ({n_w}) != rows ({len(sdf)})")
                stats[key] = w_stats

        episodes.append({"episode_index": ep, "tasks": [meta["task"]],
                         "length": len(sdf), "result": label,
                         "session": session.name, "trimmed_rows": cut,
                         "stats": stats})
        print(f"[ok]   {session.name}: {len(sdf)} rows @ {args.fps} Hz, "
              f"result={label}")
        start += len(sdf)
        ep += 1

    if rows_in and not args.no_trim:
        print(f"\ntrim total: {rows_in} rows -> {rows_kept} kept "
              f"({100 * (1 - rows_kept / rows_in):.0f}% removed)")
    if args.dry_run:
        return
    if not episodes:
        sys.exit("error: no episodes exported")
    sonic.write_meta(out, episodes, tasks, args.fps, args.chunks_size,
                     list(sonic.WRIST_KEYS) if args.wrist_cameras else [])
    print(f"wrote {len(episodes)} episode(s), "
          f"{sum(e['length'] for e in episodes)} rows -> {out}")


if __name__ == "__main__":
    main()
