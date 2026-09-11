#!/usr/bin/env python3
"""export_sonic_trim.py + observation.uwb 기록용 extra 컬럼.

UWB 절대 위치(x,y,z m, ~10 Hz)를 50 Hz 토큰 그리드에 nearest recv_ns로
붙여 observation.uwb[3]으로 parquet에 저장한다. 사전 등록된
unitree_g1_sonic 임보디먼트의 modality_keys에 없는 키라서 GR00T 학습
로더는 읽지 않는다 — 분석/디버깅용 기록일 뿐 학습에는 영향 없음.
modality.json은 건드리지 않고 info.json features에만 등록한다.

구현은 export_sonic_trim.py의 main을 그대로 쓰되(트리밍 로직 동기화
유지), 그 main이 부르는 훅을 감싼다:
  build_rows         -> uwb.csv nearest 조인 + 손가락 반응성 체크
  assemble_features  -> observation.uwb 벡터 추가
  write_meta         -> info.json에 observation.uwb feature 등록
  trim.motion_start  -> 동작 시작 감지를 4부위(왼팔/오른팔/왼손/오른손)로 확장

트리밍 (export_sonic_trim.py의 오른쪽-전용 감지를 대체):
  왼팔(m15~21)/오른팔(m22~28)은 --arm-disp(0.15 rad), 왼손/오른손(Dex3
  측정)은 --hand-disp(0.1 rad) 기준으로, **어느 부위든 시작 자세에서
  처음 벗어난 시점**을 동작 시작으로 잡고 그 앞을 pad만 남기고 절단.
  근거: 0909 수집분은 왼손 주도 태스크라 오른쪽만 감시하면 27개 중
  26개 세션에서 실제 동작이 0.3~3.3 s 잘림. 다리/허리는 감시 안 함
  (0902 수집분에 다리 선행 1건 있었음 — 필요 시 LIMBS에 추가).

손가락 체크 (세션별, 양손 f0~f6):
  [경고] 무반응 손가락: cmd q 범위 > 0.3 rad인데 측정 q 범위 < 0.05 rad
         (명령은 나가는데 관절이 안 움직임 — 하드웨어 걸림/모터 미반응,
         실제 사례 20260901_084457 왼손 f5). 기본은 경고만 하고 포함,
         --skip-bad-hands를 주면 해당 세션 제외.
  [참고] 정적 그립: 양손 14관절 전부 cmd 범위 < 0.1 rad (주먹을 미리 쥔 채
         녹화 — 0820 수집분 62/116 사례). 정보성 출력만.
  cmd가 항상 0인 f0(thumb0)은 설계상 미사용이라 걸리지 않음.

Usage: export_sonic_trim.py CLI + --skip-bad-hands.
  python export_sonic_trim_uwb.py SESSIONS_ROOT --out DATASET_DIR
      [--results success] [--wrist-cameras] [--skip-bad-hands] [--dry-run] [...]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_lerobot as base
import export_sonic as sonic
import export_sonic_trim as trim

UWB_KEY = "observation.uwb"
UWB_COLS = ["uwb.x", "uwb.y", "uwb.z"]

# 손가락 반응성 체크 임계값 (rad)
CMD_MOVE = 0.3     # 이만큼 이상 명령이 움직였는데
STATE_MOVE = 0.05  # 측정이 이 미만이면 "무반응 손가락"
STATIC_GRIP = 0.1  # 전 관절 cmd 범위가 이 미만이면 "정적 그립" 세션

SKIP_BAD_HANDS = False  # __main__에서 --skip-bad-hands로 설정

HANDS = {"왼손": ("cl", "hl"), "오른손": ("cr", "hr")}


def check_hands(df, name: str) -> bool:
    """무반응 손가락 경고 (True = --skip-bad-hands 시 세션 제외 대상)."""
    dead, cmd_ranges = [], []
    for side, (cmd, st) in HANDS.items():
        for f in sonic.HAND:
            c = df[f"{cmd}.{f}_q"]
            s = df[f"{st}.{f}_q"]
            c_rng, s_rng = c.max() - c.min(), s.max() - s.min()
            cmd_ranges.append(c_rng)
            if c_rng > CMD_MOVE and s_rng < STATE_MOVE:
                dead.append(f"{side} {f} (cmd {c_rng:.2f} rad → 측정 {s_rng:.3f} rad)")
    if dead:
        print(f"  경고: {name}: 무반응 손가락 — " + "; ".join(dead))
    elif max(cmd_ranges) < STATIC_GRIP:
        print(f"  참고: {name}: 손 명령이 세션 내내 고정 (정적 그립)")
    return bool(dead)


# 동작 시작 감지 대상: 그룹명 -> (조인된 df 컬럼, 임계값 argparse 이름)
LIMBS = {
    "왼팔": ([f"st.m{i}_q" for i in range(15, 22)], "arm_disp"),
    "오른팔": ([f"st.m{i}_q" for i in range(22, 29)], "arm_disp"),
    "왼손": ([f"hl.{f}_q" for f in sonic.HAND], "hand_disp"),
    "오른손": ([f"hr.{f}_q" for f in sonic.HAND], "hand_disp"),
}


def motion_start_all(df, fps: int, args) -> int | None:
    """4부위 중 어느 쪽이든 시작 자세를 처음 벗어난 행 (None = 아무도 안 움직임)."""
    firsts = {}
    for name, (cols, thr) in LIMBS.items():
        a = df[cols].to_numpy(dtype=np.float64)
        idx = np.flatnonzero(np.abs(a - a[0]).max(axis=1) > getattr(args, thr))
        if idx.size:
            firsts[name] = int(idx[0])
    if not firsts:
        return None
    name = min(firsts, key=firsts.get)
    print(f"  동작 시작: {name} @ {firsts[name] / fps:.1f}s")
    return firsts[name]


def build_rows_uwb(session: Path, camera: str, decoded_seqs: list):
    """sonic.build_rows + uwb(x,y,z) nearest-join (행 수/순서 불변)."""
    df, frame_seqs = _build_rows(session, camera, decoded_seqs)
    if frame_seqs is None or len(df) == 0:
        return df, frame_seqs
    if check_hands(df, session.name) and SKIP_BAD_HANDS:
        print(f"[skip] {session.name}: 무반응 손가락 (--skip-bad-hands)")
        return df.head(0), None
    df = base._asof(df, session / "uwb.csv", ["x", "y", "z"],
                    "nearest").rename(columns={c: f"uwb.{c}" for c in "xyz"})
    if df[UWB_COLS].isna().any().any():
        sys.exit(f"error: {session.name}: uwb.csv 조인에 NaN — uwb 스트림 확인")
    return df, frame_seqs


def assemble_features_uwb(df) -> dict:
    vecs = _assemble_features(df)
    vecs[UWB_KEY] = df[UWB_COLS].to_numpy(dtype=np.float64)
    return vecs


def write_meta_uwb(out: Path, *args, **kwargs):
    _write_meta(out, *args, **kwargs)
    info_path = out / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"][UWB_KEY] = {"dtype": "float64", "shape": [3],
                                 "names": ["uwb_x", "uwb_y", "uwb_z"]}
    info_path.write_text(json.dumps(info, indent=4))


_build_rows = sonic.build_rows
_assemble_features = sonic.assemble_features
_write_meta = sonic.write_meta

if __name__ == "__main__":
    if "--skip-bad-hands" in sys.argv:
        SKIP_BAD_HANDS = True
        sys.argv.remove("--skip-bad-hands")  # trim의 argparse에 넘기지 않음
    sonic.build_rows = build_rows_uwb
    sonic.assemble_features = assemble_features_uwb
    sonic.write_meta = write_meta_uwb
    trim.motion_start = motion_start_all
    trim.main()
