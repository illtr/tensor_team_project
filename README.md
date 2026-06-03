# Ultra-Fast-Lane-Detection V2 설치 및 실행 가이드

> 출처: https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2

## 목차
1. [필요 환경](#1-필요-환경)
2. [설치 방법](#2-설치-방법)
3. [데이터셋 준비](#3-데이터셋-준비)
4. [실행 방법](#4-실행-방법)
5. [결과 저장 위치](#5-결과-저장-위치)
6. [배포 (ONNX / TensorRT)](#6-배포-onnx--tensorrt)

---

## 1. 필요 환경

### 운영체제
- **Linux** 또는 **Windows + WSL2 (권장)** 필수
- 순수 Windows 환경에서는 아래 두 가지 이유로 설치가 불가합니다:
  - NVIDIA DALI가 Linux 전용 패키지 (Windows 빌드 미제공)
  - 커스텀 CUDA 확장 빌드 스크립트(`sh build.sh`)가 bash 환경 필요
- WSL2 설치: `wsl --install` (PowerShell 관리자 모드에서 실행 후 재부팅)

### 하드웨어
- NVIDIA GPU (CUDA 지원 필수)
- GPU 메모리: 최소 8GB 권장 (batch_size=32 기준)

### 소프트웨어 의존성

**Python 패키지** (`requirements.txt`):
```
opencv-python
tqdm
tensorboard
addict
scikit-learn
pathspec
imagesize
ujson
```

**추가로 필요한 패키지**:
| 패키지 | 용도 |
|---|---|
| PyTorch + torchvision | 딥러닝 프레임워크 |
| CUDA 11.x | GPU 연산 |
| nvidia-dali | 고속 데이터 로딩 |
| scipy | 곡선 피팅 (CurveLanes 평가 시) |
| my_interp (내장 CUDA 확장) | 커스텀 보간 연산 |

**CULane 평가 도구** (평가 시에만 필요):
- OpenCV C++ 라이브러리
- GCC 7.3.0 이상 (Linux) 또는 Visual Studio 2017+ (Windows)

---

## 2. 설치 방법

### 단계 1: 저장소 클론

```bash
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection-V2
cd Ultra-Fast-Lane-Detection-V2
```

### 단계 2: 가상환경 생성 및 활성화

```bash
conda create -n lane-det python=3.7 -y
conda activate lane-det
```

### 단계 3: PyTorch 설치

```bash
# CUDA 11.7 기준 (환경에 맞게 버전 조정)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 단계 4: Python 의존성 설치

```bash
pip install -r requirements.txt
```

### 단계 5: NVIDIA DALI 설치 (고속 데이터 로더)

> **Windows 사용자**: 이 단계부터는 WSL2 터미널 안에서 실행해야 합니다. DALI는 Linux 전용입니다.

```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist \
    --upgrade nvidia-dali-cuda110
```

### 단계 6: 커스텀 CUDA 확장 빌드

> **WSL2 필수**: `sh build.sh`는 bash 환경에서만 실행됩니다.

```bash
cd my_interp
sh build.sh
cd ..
```

> GCC 빌드 실패 시 GCC를 7.3.0 이상으로 업그레이드 필요

### 단계 7 (선택): CULane 평가 도구 빌드

CULane 데이터셋으로 F-measure 평가를 원할 때만 필요합니다.

**Linux:**
```bash
cd evaluation/culane
make
```

**Windows (Visual Studio 2017):**
```bash
cd evaluation/culane
mkdir build-vs2017 && cd build-vs2017
cmake .. -G "Visual Studio 15 2017 Win64"
cmake --build . --config Release
move culane_evaluator ../evaluate
```

---

## 3. 데이터셋 준비

세 가지 데이터셋을 지원합니다. 사용할 데이터셋 하나만 준비하면 됩니다.

### 3-1. TuSimple

**디렉토리 구조:**
```
$TUSIMPLE/
├── clips/
├── label_data_0313.json
├── label_data_0531.json
├── label_data_0601.json
├── test_tasks_0627.json
├── test_label.json
└── readme.md
```

**전처리** (세그멘테이션 어노테이션 및 파일 목록 생성):
```bash
python scripts/convert_tusimple.py --root /path/to/your/tusimple
# → train_gt.txt, test.txt 생성
```

### 3-2. CULane

**디렉토리 구조:**
```
$CULANE/
├── driver_100_30frame/
├── driver_161_90frame/
├── driver_182_30frame/
├── driver_193_90frame/
├── driver_23_30frame/
├── driver_37_30frame/
├── laneseg_label_w16/
└── list/
```

**전처리** (어노테이션 캐시 파일 생성, 학습 속도 향상):
```bash
python scripts/cache_culane_ponits.py --root /path/to/your/culane
# → culane_anno_cache.json 생성
```

### 3-3. CurveLanes

**디렉토리 구조:**
```
$CURVELANES/
├── test/
├── train/
└── valid/
```

**전처리:**
```bash
python scripts/convert_curvelanes.py --root /path/to/your/curvelanes
python scripts/make_curvelane_as_culane_test.py --root /path/to/your/curvelanes
# → curvelanes_anno_cache_train.json 및 .lines.txt 파일 생성
```

---

## 4. 실행 방법

모든 명령은 config 파일 경로를 첫 번째 인자로 받습니다.  
`configs/` 폴더의 파일을 먼저 열어 `data_root`와 `log_path`를 실제 경로로 수정하세요.

### 4-1. 학습 (Training)

```bash
# 단일 GPU
python train.py configs/tusimple_res18.py

# 다중 GPU (예: GPU 4개)
python -m torch.distributed.launch --nproc_per_node=4 train.py configs/culane_res18.py
```

**주요 config 파라미터 (명령행에서 오버라이드 가능):**

```bash
python train.py configs/tusimple_res18.py \
    --data_root /path/to/tusimple \
    --log_path /path/to/log \
    --batch_size 16 \
    --epoch 100
```

| 파라미터 | 기본값 (tusimple_res18) | 설명 |
|---|---|---|
| `backbone` | `18` | ResNet 깊이 (18/34/50/101/152) |
| `batch_size` | `32` | 배치 크기 |
| `epoch` | `100` | 총 학습 에폭 |
| `learning_rate` | `0.05` | 초기 학습률 |
| `use_aux` | `False` | 보조 세그멘테이션 헤드 사용 여부 |
| `num_lanes` | `4` | 검출할 차선 수 |
| `train_width` | `800` | 입력 이미지 너비 |
| `train_height` | `320` | 입력 이미지 높이 |
| `finetune` | `None` | 파인튜닝할 모델 경로 |
| `resume` | `None` | 이어서 학습할 체크포인트 경로 |

### 4-2. 테스트 / 평가 (Testing)

```bash
python test.py configs/tusimple_res18.py \
    --test_model /path/to/model_best.pth \
    --test_work_dir /path/to/output
```

### 4-3. 데모 영상 생성 (Demo)

학습된 모델로 테스트 이미지에 차선을 시각화한 AVI 영상을 생성합니다.

```bash
python demo.py configs/tusimple_res18.py \
    --test_model /path/to/model_best.pth \
    --data_root /path/to/tusimple
```

### 4-4. 속도 측정

```bash
python speed_simple.py
```

---

## 5. 결과 저장 위치

### 학습 결과

학습이 시작되면 `log_path` 아래에 타임스탬프 기반 디렉토리가 자동 생성됩니다.

```
{log_path}/
└── 20240101_120000_lr_5e-02_b_32{note}/   ← 자동 생성 (날짜_시간_lr_배치)
    ├── model_best.pth                       ← 검증 F1이 가장 높을 때 저장
    ├── cfg.txt                              ← 사용된 설정값 전체 기록
    └── events.out.tfevents.*               ← TensorBoard 로그
```

**TensorBoard로 학습 곡선 확인:**
```bash
tensorboard --logdir /path/to/log
```

모니터링 가능한 지표:
- `loss/cls_loss` : 위치 분류 손실 (행/열)
- `loss/relation_loss` : 차선 연속성 손실
- `loss/mean_loss_row` / `loss/mean_loss_col` : 평균 위치 정규화 손실
- `metric/top1`, `metric/top2`, `metric/top3` : 위치 예측 정확도
- `metric/ext_row`, `metric/ext_col` : 차선 존재 여부 정확도
- `CuEval/X` : 에폭별 최고 F-measure (학습 중 검증)

### 평가 결과

`test_work_dir` 아래에 데이터셋별로 생성됩니다.

**CULane:**
```
{test_work_dir}/
├── culane_eval_tmp/              ← 이미지별 예측 좌표 (.lines.txt)
│   └── driver_.../
│       └── *.lines.txt
└── txt/
    ├── out0_normal.txt           ← 시나리오별 TP/FP/FN/F-measure
    ├── out1_crowd.txt
    ├── out2_hlight.txt
    ├── out3_shadow.txt
    ├── out4_noline.txt
    ├── out5_arrow.txt
    ├── out6_curve.txt
    ├── out7_cross.txt
    └── out8_night.txt
```

**TuSimple:**
```
{test_work_dir}/
└── tusimple_eval_tmp.txt         ← JSON Lines 형식 (이미지별 차선 좌표)
```

**데모 결과:**
```
{실행 디렉토리}/
├── test0_normal.avi
├── test1_crowd.avi
└── ...                           ← 각 테스트 분류별 AVI 영상
```

### .lines.txt 파일 형식

각 줄이 차선 하나를 의미하며 `x1 y1 x2 y2 ...` 형식으로 좌표쌍이 나열됩니다.

```
832.500 248.000 810.500 278.000 798.500 308.000 ...
1012.500 248.000 995.500 278.000 ...
```

---

