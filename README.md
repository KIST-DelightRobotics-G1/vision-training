# kist-vision-training

kist-data-collector 세션(무손실 원본 녹화)을 GR00T N1.7 파인튜닝용 LeRobot
데이터셋으로 변환한다.

**주 사용 스크립트: `src/preprocess/export_sonic_trim_uwb.py`**
(`--embodiment-tag UNITREE_G1_SONIC`용 데이터셋 + 앞부분 대기 구간 트리밍 +
손가락 무반응 체크 + UWB 위치 기록 컬럼)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

시스템에 libx264가 포함된 `ffmpeg`가 필요하다. lerobot 라이브러리는 필요 없음.

## Usage

```bash
python3 src/preprocess/export_sonic_trim_uwb.py <sessions-root> --out <dataset-dir>

# 일반적인 사용: 성공 에피소드만, 손목 카메라 포함
python3 src/preprocess/export_sonic_trim_uwb.py ~/sessions_0917 --out ~/sonic_train \
    --results success --wrist-cameras

# 먼저 세션별 트리밍/스킵 결과만 확인 (아무것도 쓰지 않음)
python3 src/preprocess/export_sonic_trim_uwb.py ~/sessions_0917 --out ~/sonic_train \
    --results success --dry-run
```

`<sessions-root>`는 세션 디렉터리(`meta.yaml` 포함) 자체이거나 그 상위
디렉터리이며, 여러 개를 나열할 수 있다. 출력 디렉터리는 비어 있어야 한다.

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--results` | `all` | `success,fail,unlabeled` 중 포함할 라벨 (라벨은 어느 경우든 `episodes.jsonl`에 기록) |
| `--wrist-cameras` | off | `left_wrist` / `right_wrist` 영상도 함께 저장 |
| `--camera` | `head` | ego_view 영상 소스 카메라 |
| `--fps` | `50` | 모션 토큰 그리드 주기 (Hz) |
| `--min-rows` | `50` | 트리밍 후 이보다 짧은 에피소드 제외 (50 = 1 s, VLA 청크는 40 필요) |
| `--min-cam-fps` | `25` | 헤드 카메라 수신율이 이보다 낮은 세션 제외 (0 = 끔) |
| `--arm-disp` | `0.15` | 팔/다리·허리 동작 시작 판정 변위 (rad) |
| `--hand-disp` | `0.1` | 손(Dex3) 동작 시작 판정 변위 (rad) |
| `--pad` | `0.5` | 동작 시작 앞에 남겨둘 정지 구간 (s) |
| `--no-trim` | off | 트리밍 비활성화 (전체 행 유지, 동작 미검출 스킵도 안 함) |
| `--skip-bad-hands` | off | 무반응 손가락이 있는 세션 제외 (기본은 경고만) |
| `--dry-run` | off | 세션별 트리밍 결과만 출력하고 아무것도 쓰지 않음 |

## export_sonic_trim_uwb.py가 하는 일

1. **세션 필터링**: 라벨(`--results`), 카메라 스트림 유무, 헤드 카메라
   수신율(`--min-cam-fps`)로 거른다.
2. **50 Hz 정렬**: `motion_token.csv` 틱마다 lowstate / 손 측정·명령 /
   lowcmd / 카메라 프레임을 recv_ns 기준으로 조인한다 (`export_sonic.py`).
3. **손가락 체크** (양손 f0~f6):
   - 경고 — 무반응 손가락: cmd 범위 > 0.3 rad인데 측정 범위 < 0.05 rad
     (명령은 나가는데 관절이 안 움직임, 예: `20260901_084457` 왼손 f5).
     `--skip-bad-hands`면 세션 제외.
   - 참고 — 정적 그립: 양손 14관절 모두 cmd 범위 < 0.1 rad (주먹을 쥔 채 녹화).
4. **앞부분 대기 구간 트리밍**: 5부위(왼팔 m15~21, 오른팔 m22~28,
   다리·허리 m00~14, 왼손, 오른손) 중 **어느 부위든 시작 자세에서 처음
   벗어난 시점**을 동작 시작으로 잡고, 그 앞을 `--pad`만 남기고 자른다.
   속도가 아니라 첫 행 대비 변위로 판정한다 (텔레옵 떨림 때문에 속도
   임계값은 오작동). 동작 시작 이후는 건드리지 않는다. 어느 부위도
   움직이지 않은 세션은 제외.
5. **UWB 조인**: `uwb.csv`의 x,y,z (m, ~10 Hz)를 nearest로 붙여
   `observation.uwb[3]`로 저장한다. `uwb.csv`가 비어 있는 세션은 스키마를
   맞추기 위해 제외(경고 출력).

`observation.uwb`는 `info.json` features에만 등록되고 `modality.json`에는
없으므로 GR00T 학습 로더는 읽지 않는다 — 분석/디버깅용 기록일 뿐이다.

## Output structure

```
<dataset-dir>/
├── data/chunk-000/
│   └── episode_000000.parquet   # 50 Hz 모션 토큰 틱당 한 행:
│                                #   observation.state[43], action.motion_token[64],
│                                #   action.wbc[43], projected_gravity, root quat,
│                                #   teleop hand joints, observation.uwb[3],
│                                #   timestamp, indices
├── videos/chunk-000/
│   ├── observation.images.ego_view/episode_000000.mp4      # h264, 50 fps
│   ├── observation.images.left_wrist/...                   # --wrist-cameras
│   └── observation.images.right_wrist/...
└── meta/
    ├── info.json                # features, fps, splits, path templates
    ├── modality.json            # GR00T group slices (UNITREE_G1_SONIC contract)
    ├── tasks.jsonl              # task_index <-> language instruction
    ├── episodes.jsonl           # 에피소드별 length, task, result, session, trimmed_rows
    └── episodes_stats.jsonl     # 에피소드별 min/max/mean/std
```

이후 Isaac-GR00T 환경에서 파인튜닝:

```bash
uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path <dataset-dir> --embodiment-tag UNITREE_G1_SONIC ...
```

## 기타 스크립트 (src/preprocess/)

`export_sonic_trim_uwb.py`는 아래 모듈을 import해서 훅만 바꿔 끼우는
구조라 함께 유지해야 한다.

- `export_sonic.py` — 기본 변환기 (트리밍/UWB 없음). 50 Hz 정렬, parquet/영상/메타 작성
- `export_sonic_trim.py` — export_sonic + 트리밍 메인 루프 (오른팔/오른손만 감시하는 구버전 감지; uwb 버전이 5부위 감지로 교체)
- `export_lerobot.py` — Humanoid-Everyday-G1 스키마 변환기 겸 공용 정렬/세션 탐색 모듈
- `dataset_viewer.py` / `view_parquet.py` — 검사용: 에피소드별 HTML 뷰어(영상 + 라벨 행), parquet 덤프/비교
