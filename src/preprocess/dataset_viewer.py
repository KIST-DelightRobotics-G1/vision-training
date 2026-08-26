#!/usr/bin/env python3
"""LeRobot 데이터셋 HTML 뷰어 생성 — HF Data Studio 스타일.

에피소드별로: ego/wrist 영상 재생 + 전 행(row) 테이블. 벡터 컬럼은
meta/info.json 이름 + meta/modality.json 그룹으로 레이블링되고,
그룹 헤더로 묶여서 표시됨. 영상은 상대경로 참조 (데이터셋 폴더 안에
HTML을 만들므로 브라우저로 열면 바로 재생).

  python dataset_viewer.py ~/sonic_train_0820_trim2 -e 0
  # -> ~/sonic_train_0820_trim2/viewer_episode_000000.html
  firefox ~/sonic_train_0820_trim2/viewer_episode_000000.html

옵션: --max-rows N (기본 전부), --episodes 0 1 2 (여러 개 생성)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from view_parquet import build_labels, expand, find_meta

CSS = """
body{font:13px/1.4 system-ui,sans-serif;margin:0;background:#111;color:#ddd}
header{padding:12px 16px;background:#1a1a2e;position:sticky;top:0;z-index:3}
h1{font-size:15px;margin:0 0 4px}
.meta{color:#9aa;font-size:12px}
.videos{display:flex;gap:8px;padding:8px 16px;background:#161625}
.videos figure{margin:0}
.videos video{width:320px;border:1px solid #333;border-radius:4px}
.videos figcaption{font-size:11px;color:#9aa;text-align:center}
.wrap{overflow:auto;max-height:70vh}
table{border-collapse:collapse;font-size:11.5px;white-space:nowrap}
th,td{border:1px solid #2a2a3a;padding:2px 7px;text-align:right}
thead th{position:sticky;background:#22223a;color:#cfe;z-index:2}
thead tr:first-child th{top:0}
thead tr:last-child th{top:26px;font-weight:400;color:#9ac}
tbody th{position:sticky;left:0;background:#1a1a2e;color:#fc6}
tr:hover td{background:#20304a}
.g0{background:#181828}.g1{background:#141420}
td.diffpos{color:#7f7}td.diffneg{color:#f88}
input{margin:8px 16px;padding:4px 8px;background:#222;color:#ddd;
      border:1px solid #444;border-radius:4px;width:260px}
"""

JS = """
document.getElementById('filter').addEventListener('input', e=>{
  const q=e.target.value.toLowerCase();
  document.querySelectorAll('col').forEach(()=>{});
  const table=document.getElementById('t');
  const groups=[...table.tHead.rows[0].cells];
  const labs=[...table.tHead.rows[1].cells];
  labs.forEach((th,i)=>{
    if(i===0)return;
    const show=!q||th.textContent.toLowerCase().includes(q)
      ||th.dataset.g.toLowerCase().includes(q);
    table.querySelectorAll(`[data-c="${i}"]`).forEach(c=>
      c.style.display=show?'':'none');
  });
});
"""


def group_of(colname: str) -> str:
    lab = colname.split("::")[-1]
    return lab.split(".")[0] if "." in lab else colname.split("::")[0]


def make_html(root: Path, ep: int, max_rows: int | None) -> Path:
    meta = find_meta(root)
    info = json.loads((meta / "info.json").read_text())
    hits = sorted(root.glob(f"data/*/episode_{ep:06d}.parquet"))
    if not hits:
        raise SystemExit(f"error: episode {ep} 없음")
    flat = expand(pd.read_parquet(hits[0]), build_labels(meta))
    n_all = len(flat)
    if max_rows:
        flat = flat.iloc[:max_rows]

    videos = sorted(root.glob(f"videos/*/*/episode_{ep:06d}.mp4"))
    task = "?"
    for line in (meta / "episodes.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["episode_index"] == ep:
            task = e["tasks"][0]
            break

    cols = [c for c in flat.columns]
    groups, spans = [], []
    for c in cols:
        g = group_of(c)
        if groups and groups[-1] == g:
            spans[-1] += 1
        else:
            groups.append(g)
            spans.append(1)

    h = [f"<!-- viewer --><title>ep{ep:06d} — {root.name}</title>",
         f"<style>{CSS}</style>",
         f"<header><h1>{root.name} / episode_{ep:06d}</h1>",
         f"<div class='meta'>task: {task} · rows {n_all} · fps {info['fps']}"
         f" · {len(cols)} labeled columns</div></header>",
         "<div class='videos'>"]
    for v in videos:
        cam = v.parent.name.replace("observation.images.", "")
        rel = v.relative_to(root)
        h.append(f"<figure><video src='{rel}' controls muted loop></video>"
                 f"<figcaption>{cam}</figcaption></figure>")
    h.append("</div>")
    h.append("<input id='filter' placeholder='컬럼 필터 (예: right_arm, motion_token)'>")
    h.append("<div class='wrap'><table id='t'><thead><tr><th></th>")
    gi = 0
    for g, s in zip(groups, spans):
        h.append(f"<th colspan='{s}' class='g{gi % 2}'>{g}</th>")
        gi += 1
    h.append("</tr><tr><th>row</th>")
    gmap = {}
    gi = 0
    ci = 1
    for g, s in zip(groups, spans):
        for _ in range(s):
            gmap[ci] = (g, gi % 2)
            ci += 1
        gi += 1
    for i, c in enumerate(cols, 1):
        lab = c.split("::")[-1].split(".")[-1]
        h.append(f"<th data-g='{gmap[i][0]}' data-c='{i}' "
                 f"class='g{gmap[i][1]}'>{lab}</th>")
    h.append("</tr></thead><tbody>")
    vals = flat.to_numpy()
    for r in range(len(flat)):
        h.append(f"<tr><th>{flat.index[r]}</th>")
        row = vals[r]
        cells = []
        for i, v in enumerate(row, 1):
            s = f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
            cells.append(f"<td data-c='{i}' class='g{gmap[i][1]}'>{s}</td>")
        h.append("".join(cells) + "</tr>")
    h.append(f"</tbody></table></div><script>{JS}</script>")

    out = root / f"viewer_episode_{ep:06d}.html"
    out.write_text("\n".join(h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("-e", "--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    root = Path(args.dataset).expanduser()
    for ep in args.episodes:
        out = make_html(root, ep, args.max_rows)
        print(f"생성: {out}")


if __name__ == "__main__":
    main()
