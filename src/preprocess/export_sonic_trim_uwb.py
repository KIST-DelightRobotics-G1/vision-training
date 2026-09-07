#!/usr/bin/env python3
"""export_sonic_trim.py + observation.uwb 기록용 extra 컬럼.

UWB 절대 위치(x,y,z m, ~10 Hz)를 50 Hz 토큰 그리드에 nearest recv_ns로
붙여 observation.uwb[3]으로 parquet에 저장한다. 사전 등록된
unitree_g1_sonic 임보디먼트의 modality_keys에 없는 키라서 GR00T 학습
로더는 읽지 않는다 — 분석/디버깅용 기록일 뿐 학습에는 영향 없음.
modality.json은 건드리지 않고 info.json features에만 등록한다.

구현은 export_sonic_trim.py의 main을 그대로 쓰되(트리밍 로직 동기화
유지), 그 main이 부르는 sonic 훅 세 개만 감싼다:
  build_rows         -> uwb.csv nearest 조인 추가
  assemble_features  -> observation.uwb 벡터 추가
  write_meta         -> info.json에 observation.uwb feature 등록

Usage: export_sonic_trim.py와 CLI 동일.
  python export_sonic_trim_uwb.py SESSIONS_ROOT --out DATASET_DIR
      [--results success] [--wrist-cameras] [--dry-run] [...]
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


def build_rows_uwb(session: Path, camera: str, decoded_seqs: list):
    """sonic.build_rows + uwb(x,y,z) nearest-join (행 수/순서 불변)."""
    df, frame_seqs = _build_rows(session, camera, decoded_seqs)
    if frame_seqs is None or len(df) == 0:
        return df, frame_seqs
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
    sonic.build_rows = build_rows_uwb
    sonic.assemble_features = assemble_features_uwb
    sonic.write_meta = write_meta_uwb
    trim.main()
