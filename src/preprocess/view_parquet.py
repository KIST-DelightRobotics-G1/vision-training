#!/usr/bin/env python3
"""LeRobot 파케이 뷰어 — 벡터 컬럼을 관절/그룹 레이블로 풀어서 보여줌.

레이블 출처: 데이터셋 meta/info.json(names) + meta/modality.json(그룹 슬라이스).
그래서 43차원 state가 'right_arm.right_elbow_joint = 0.312' 식으로 나옴.

사용 예 (kist-vision-training/.venv 또는 pandas 있는 아무 환경):
  # 에피소드 0의 100번째 행을 그룹 레이블로 출력
  python view_parquet.py ~/sonic_train_0820_trim2 --episode 0 --row 100

  # 특정 그룹만, 행 범위로
  python view_parquet.py ~/sonic_train_0820_trim2 -e 0 --rows 0 200 --step 50 \
      --groups right_arm right_hand

  # 두 데이터셋의 같은 에피소드 비교 (레이블별 차이)
  python view_parquet.py ~/sonic_train_0820 -e 0 --row 50 \
      --compare ~/sonic_train_0820_trim2

  # 파케이 파일을 직접 지정해도 됨 (meta는 상위 폴더에서 자동 탐색)
  python view_parquet.py ~/sonic_train_0820/data/chunk-000/episode_000003.parquet --row 0

  # CSV로 덤프 (전 행, 레이블 헤더)
  python view_parquet.py ~/sonic_train_0820_trim2 -e 0 --csv /tmp/ep0.csv
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def find_meta(path: Path) -> Path:
    """dataset root의 meta/ 탐색 (파케이 직접 지정 시 상위로 올라감)."""
    p = path if path.is_dir() else path.parent
    for _ in range(5):
        if (p / "meta" / "info.json").is_file():
            return p / "meta"
        p = p.parent
    sys.exit(f"error: meta/info.json을 찾을 수 없음 (기준: {path})")


def build_labels(meta: Path) -> dict[str, list[str]]:
    """컬럼 -> 차원별 레이블 리스트. modality 그룹이 있으면 'group.name' 형태."""
    info = json.loads((meta / "info.json").read_text())
    modality = json.loads((meta / "modality.json").read_text()) \
        if (meta / "modality.json").is_file() else {}

    # original_key -> [(group, start, end)]  (state/action 그룹 슬라이스)
    slices: dict[str, list] = {}
    for section in ("state", "action"):
        for group, spec in modality.get(section, {}).items():
            key = spec.get("original_key",
                           "observation.state" if section == "state" else "action")
            slices.setdefault(key, []).append(
                (group, spec["start"], spec["end"]))

    labels = {}
    for col, feat in info["features"].items():
        if feat["dtype"] == "video":
            continue
        dim = feat["shape"][0]
        names = feat.get("names")
        base = (list(names) if isinstance(names, list) and len(names) == dim
                else [f"{names or col}_{i}" for i in range(dim)])
        out = list(base)
        for group, s, e in sorted(slices.get(col, []), key=lambda x: x[1]):
            for i in range(s, min(e, dim)):
                out[i] = f"{group}.{base[i]}"
        labels[col] = out
    return labels


def expand(df: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """벡터 컬럼을 레이블 붙은 스칼라 컬럼으로 펼친 flat DataFrame."""
    cols = {}
    for col in df.columns:
        v = df[col].to_numpy()
        if v.dtype == object:                      # list/array 컬럼
            arr = np.stack(v)
            for i, lab in enumerate(labels.get(col, [f"{col}_{i}" for i in
                                                     range(arr.shape[1])])):
                cols[f"{col}::{lab}"] = arr[:, i]
        else:
            cols[col] = v
    return pd.DataFrame(cols, index=df.index)


def print_row(flat: pd.DataFrame, row: int, groups, compare=None):
    r = flat.iloc[row]
    print(f"\n===== row {row}"
          + (f"  (timestamp {r.get('timestamp', float('nan')):.2f}s)" if "timestamp" in r else "")
          + " =====")
    cur_group = None
    for name, val in r.items():
        lab = name.split("::")[-1]
        grp = lab.split(".")[0] if "." in lab else name.split("::")[0]
        if groups and not any(g in name for g in groups):
            continue
        if grp != cur_group:
            print(f"\n[{grp}]")
            cur_group = grp
        line = f"  {lab:42s} {val: .4f}" if isinstance(val, (int, float, np.floating)) \
            else f"  {lab:42s} {val}"
        if compare is not None and name in compare.columns and row < len(compare):
            other = compare.iloc[row][name]
            if isinstance(val, (int, float, np.floating)):
                d = val - other
                mark = "   <-- DIFF" if abs(d) > 1e-9 else ""
                line += f"   | 비교값 {other: .4f}  (차 {d:+.4f}){mark}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="데이터셋 루트 또는 .parquet 파일")
    ap.add_argument("-e", "--episode", type=int, default=0)
    ap.add_argument("--row", type=int, help="한 행 출력")
    ap.add_argument("--rows", type=int, nargs=2, metavar=("START", "END"),
                    help="행 범위 출력")
    ap.add_argument("--step", type=int, default=1, help="--rows 간격")
    ap.add_argument("--groups", nargs="+",
                    help="이 문자열이 포함된 그룹/컬럼만 (예: right_arm motion_token)")
    ap.add_argument("--compare", help="비교할 다른 데이터셋 루트 (같은 에피소드)")
    ap.add_argument("--csv", help="flat 테이블을 CSV로 저장")
    ap.add_argument("--summary", action="store_true",
                    help="행 출력 대신 컬럼별 min/max/mean 요약")
    args = ap.parse_args()

    path = Path(args.path).expanduser()
    if path.suffix == ".parquet":
        pq_path, meta = path, find_meta(path)
    else:
        meta = find_meta(path)
        hits = sorted(path.glob(f"data/*/episode_{args.episode:06d}.parquet"))
        if not hits:
            sys.exit(f"error: episode {args.episode} 파케이 없음: {path}/data/")
        pq_path = hits[0]

    labels = build_labels(meta)
    flat = expand(pd.read_parquet(pq_path), labels)
    print(f"파일: {pq_path}\n행 {len(flat)} x 컬럼 {len(flat.columns)} (레이블 확장 후)")

    comp = None
    if args.compare:
        croot = Path(args.compare).expanduser()
        chits = sorted(croot.glob(f"data/*/episode_{args.episode:06d}.parquet"))
        if not chits:
            sys.exit(f"error: 비교 데이터셋에 episode {args.episode} 없음")
        comp = expand(pd.read_parquet(chits[0]), build_labels(find_meta(croot)))
        print(f"비교: {chits[0]} (행 {len(comp)})")

    if args.csv:
        flat.to_csv(args.csv, index=False)
        print(f"CSV 저장: {args.csv}")
        return
    if args.summary:
        desc = flat.describe().T[["min", "max", "mean", "std"]]
        if args.groups:
            desc = desc[desc.index.str.contains("|".join(args.groups))]
        with pd.option_context("display.max_rows", None,
                               "display.float_format", "{: .4f}".format):
            print(desc)
        return

    rows = ([args.row] if args.row is not None
            else range(args.rows[0], min(args.rows[1], len(flat)), args.step)
            if args.rows else [0])
    for r in rows:
        if r >= len(flat):
            print(f"(row {r}: 범위 밖, 총 {len(flat)}행)")
            break
        print_row(flat, r, args.groups, comp)


if __name__ == "__main__":
    main()
