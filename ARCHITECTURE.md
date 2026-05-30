# Ultra-Fast-Lane-Detection V2 작동 구조 설명

## 목차
1. [핵심 아이디어](#1-핵심-아이디어)
2. [전체 데이터 흐름](#2-전체-데이터-흐름)
3. [모델 구조 상세](#3-모델-구조-상세)
4. [손실 함수 구성](#4-손실-함수-구성)
5. [학습 파이프라인](#5-학습-파이프라인)
6. [추론 및 후처리](#6-추론-및-후처리)
7. [평가 방식](#7-평가-방식)
8. [파일별 역할 요약](#8-파일별-역할-요약)

---

## 1. 핵심 아이디어

### 격자 기반 위치 예측 개념

```
이미지를 수평으로 N개 구간(row anchor)으로 나눔
각 구간에서 "차선이 가로 방향 100개 셀 중 몇 번째 셀에 있는가?"를 분류

예시 (Tusimple 기준):
- 56개 수평선 (row anchor: 이미지 높이의 160~710 픽셀)
- 각 수평선에서 100개 가로 격자 중 위치 선택
- 4개 차선에 대해 독립적으로 예측

→ 픽셀 단위 예측 대신 셀 번호 예측으로 계산량 대폭 감소
```

---

## 2. 전체 데이터 흐름

```
┌─────────────────────────────────────────────────────────┐
│                      입력 이미지                         │
│                  (800 × 320, RGB)                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               Backbone: ResNet-18/34/50/...              │
│  conv1 → bn1 → relu → maxpool                           │
│  → layer1 → layer2(x2) → layer3(x3) → layer4(x4)       │
│                                                          │
│  출력: x2, x3, x4 (다중 스케일 feature map)             │
└──────────────────────┬──────────────────────────────────┘
                       │ x4 (최종 feature)
                       ▼
┌─────────────────────────────────────────────────────────┐
│              1×1 Convolution (채널 축소)                 │
│              512채널 → 8채널 (ResNet-18/34 기준)         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Flatten + MLP 분류기                    │
│  Flatten: (8 × H/32 × W/32) 개 값                      │
│  Linear(input_dim → 2048) → ReLU                        │
│  Linear(2048 → total_dim)                               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    4가지 출력 분리                        │
│                                                          │
│  loc_row   [B, 100, 56, 4]  행 방향 차선 가로 위치       │
│  loc_col   [B, 100, 41, 4]  열 방향 차선 세로 위치       │
│  exist_row [B,   2, 56, 4]  행 방향 차선 존재 여부       │
│  exist_col [B,   2, 41, 4]  열 방향 차선 존재 여부       │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   [학습 시]                   [추론 시]
   손실 계산 & 역전파          후처리: 격자 좌표 → 픽셀 좌표
                               차선 시각화 / 파일 저장
```

---

## 3. 모델 구조 상세

### 3-1. Backbone ([model/backbone.py](model/backbone.py))

ImageNet pretrained ResNet을 사용합니다. `forward()`에서 세 레벨의 feature map을 반환합니다.

```python
def forward(self, x):
    x = conv1 → bn1 → relu → maxpool → layer1
    x2 = layer2(x)   # 1/8 해상도
    x3 = layer3(x2)  # 1/16 해상도
    x4 = layer4(x3)  # 1/32 해상도
    return x2, x3, x4
```

지원 백본: `18`, `34`, `50`, `101`, `152`, `50next`, `101next`, `50wide`, `101wide`, `34fca`

### 3-2. 핵심 모델 parsingNet ([model/model_culane.py](model/model_culane.py))

```
입력 이미지 (B, 3, H, W)
    ↓ backbone
x2, x3, x4
    ↓ pool (1×1 Conv: 512ch→8ch)
fea (B, 8, H/32, W/32)
    ↓ flatten
fea (B, 8 × H/32 × W/32)
    ↓ LayerNorm (fc_norm=True일 때) → Linear → ReLU → Linear
out (B, total_dim)
    ↓ 분리
loc_row   (B, num_grid_row=100, num_cls_row=56,  num_lane=4)
loc_col   (B, num_grid_col=100, num_cls_col=41,  num_lane=4)
exist_row (B, 2,                num_cls_row=56,  num_lane=4)
exist_col (B, 2,                num_cls_col=41,  num_lane=4)
```

`total_dim = (100×56×4) + (100×41×4) + (2×56×4) + (2×41×4) = 22400 + 16400 + 448 + 328 = 39576`

### 3-3. 보조 세그멘테이션 헤드 ([model/seg_model.py](model/seg_model.py))

`use_aux=True`일 때 활성화됩니다. `x2`, `x3`, `x4` feature를 합쳐 픽셀 단위 세그멘테이션 마스크를 추가로 출력하여 학습을 보조합니다. 추론 시에는 주 출력만 사용합니다.

### 3-4. TTA (Test Time Augmentation)

`forward_tta()` 메서드는 원본 이미지 외에 좌/우 이동, 상/하 이동된 feature를 추가로 만들어 5개의 예측을 앙상블하여 정확도를 높입니다.

---

## 4. 손실 함수 구성

([utils/loss.py](utils/loss.py), [utils/factory.py](utils/factory.py))

학습 시 여러 손실을 가중합산하여 최종 손실을 계산합니다.

### TuSimple / CULane 기준 손실 구성

| 손실 이름 | 클래스 | 가중치 | 역할 |
|---|---|---|---|
| `cls_loss` | `SoftmaxFocalLoss` | 1.0 | 행 방향 위치 분류 (핵심 손실) |
| `cls_loss_col` | `SoftmaxFocalLoss` | 1.0 | 열 방향 위치 분류 |
| `cls_ext` | `CrossEntropyLoss` | 1.0 | 행 방향 차선 존재 여부 분류 |
| `cls_ext_col` | `CrossEntropyLoss` | 1.0 | 열 방향 차선 존재 여부 분류 |
| `relation_loss` | `ParsingRelationLoss` | `sim_loss_w` | 인접 행 간 예측값 연속성 유지 |
| `relation_dis` | `ParsingRelationDis` | `shp_loss_w` | 차선 형태의 전체적인 형상 일관성 |
| `mean_loss_row` | `MeanLoss` | 0.05 | 예측 분포의 평균이 정답에 가깝도록 |
| `mean_loss_col` | `MeanLoss` | 0.05 | 열 방향 동일 |

### 손실 계산 방식 시각화

```
[loc_row 예측] (softmax 분포)
  셀 번호:  0   1   2  ...  99
  확률:    0.01 0.02 0.9 ...  0.0   ← 99번 셀이 높음 → 차선이 오른쪽에 있음

[SoftmaxFocalLoss] = Focal(1-p)^γ × NLL
  → 틀린 예측에 더 큰 패널티, soft label로 인접 셀에도 약간의 정답 배분

[MeanLoss]
  예측 평균 위치 = Σ(확률 × 셀번호) → 이것이 정답 셀과 가까워지도록

[ParsingRelationLoss]
  인접 행들의 예측 분포 차이 최소화 → 차선이 부드럽게 연결되도록
```

---

## 5. 학습 파이프라인

([train.py](train.py), [utils/common.py](utils/common.py))

```
train.py 실행
    │
    ├─ merge_config(): config 파일 + 명령행 인자 병합
    ├─ get_train_loader(): DALI 기반 고속 데이터 로더 생성
    ├─ get_model(): 데이터셋에 맞는 모델 인스턴스화
    ├─ get_optimizer(): SGD 또는 Adam
    ├─ get_scheduler(): MultiStepLR 또는 CosineAnnealingLR (linear warmup 포함)
    └─ get_loss_dict() / get_metric_dict()
    
    ┌─ for epoch in range(epochs):
    │    ┌─ for batch in train_loader:
    │    │    ├─ inference(): 모델 순전파
    │    │    ├─ calc_loss(): 각 손실 계산 및 가중합산
    │    │    ├─ loss.backward() + optimizer.step()
    │    │    └─ (20 step마다) TensorBoard에 손실/메트릭 기록
    │    │
    │    └─ eval_lane(): 에폭 끝마다 검증 데이터셋으로 F-measure 계산
    │         └─ 최고 F-measure 갱신 시 model_best.pth 저장
    └─
```

### 분산 학습 지원

`WORLD_SIZE` 환경변수가 설정되면 자동으로 `DistributedDataParallel` 모드로 전환됩니다. NCCL 백엔드를 사용하며, 평가 및 모델 저장은 rank=0 프로세스에서만 수행됩니다.

### 학습률 스케줄러

**MultiStepLR** (기본값):
```
epoch 0~49: lr = 0.05
epoch 50~74: lr = 0.05 × 0.1 = 0.005
epoch 75~99: lr = 0.05 × 0.01 = 0.0005
+ warmup_iters=100 스텝 동안 선형 증가
```

---

## 6. 추론 및 후처리

([demo.py](demo.py), [evaluation/eval_wrapper.py](evaluation/eval_wrapper.py))

### 격자 인덱스 → 픽셀 좌표 변환

모델 출력은 "100개 셀 중 몇 번째"라는 정수 인덱스가 아닌 softmax 확률 분포입니다.  
후처리는 **확률 가중 평균(Soft-argmax)**으로 연속적인 좌표를 얻습니다.

```python
# 행 방향 좌표 계산 예시
all_ind = [argmax-1, argmax, argmax+1]          # 주변 셀 포함
prob = softmax(loc_row[all_ind])                # 국소 softmax
cell_pos = (prob * all_ind).sum() + 0.5         # 가중 평균 (0~100)
pixel_x = cell_pos / (num_grid - 1) * image_width  # 픽셀 좌표로 변환
pixel_y = row_anchor[k] * image_height              # 행 anchor → 픽셀 y좌표
```

### 차선 유효성 판단

- **행 방향 차선**: 전체 행 중 절반(>50%) 이상에서 `exist_row=1`이어야 유효
- **열 방향 차선**: 전체 열 중 25% 이상에서 `exist_col=1`이어야 유효

### 차선 인덱스 배정 (CULane 기준)

```
차선 인덱스 0: 가장 왼쪽 (열 방향으로 검출)
차선 인덱스 1: 왼쪽 내측 (행 방향으로 검출)  ← 자차 기준 왼쪽 차선
차선 인덱스 2: 오른쪽 내측 (행 방향으로 검출) ← 자차 기준 오른쪽 차선
차선 인덱스 3: 가장 오른쪽 (열 방향으로 검출)
```

### CurveLanes 곡선 보정

행+열 방향 예측을 합친 뒤 최소자승법(leastsq)으로 2차 다항식 곡선을 피팅하여 최종 좌표를 보정합니다.

---

## 7. 평가 방식

### TuSimple

평가 지표: **Accuracy, FP, FN, F1**

예측 결과를 JSON Lines 형식으로 저장 후 `LaneEval.bench_one_submit()`으로 공식 평가합니다.

```json
{"lanes": [[832, -2, 810, ...], [1012, ...]], "h_samples": [160, 170, ...], "raw_file": "clips/..."}
```

차선 좌표는 `h_samples`의 각 y좌표에 대응하는 x좌표 목록이며, `-2`는 해당 y에서 차선 없음을 의미합니다.

### CULane

평가 지표: **F-measure (TP/(TP+FP), TP/(TP+FN) 기반)**

C++로 작성된 평가 바이너리(`evaluation/culane/evaluate`)를 실행합니다.  
예측 좌표를 IoU 0.5 기준으로 정답과 매칭하여 9가지 시나리오별 F-measure를 산출합니다.

| 시나리오 | 설명 |
|---|---|
| Normal | 일반 도로 |
| Crowd | 차량 혼잡 |
| Highlight | 역광/강한 빛 |
| Shadow | 그늘 |
| No line | 차선 마모/소실 |
| Arrow | 화살표 표시 |
| Curve | 곡선 구간 |
| Cross | 교차로 |
| Night | 야간 |

### CurveLanes

CULane 평가 도구를 재활용하며, IoU 기준을 0.5로 설정하고 이미지 해상도에 맞게 스케일 팩터를 조정합니다.

---

## 8. 파일별 역할 요약

```
Ultra-Fast-Lane-Detection-V2/
│
├── train.py               학습 진입점, 에폭 루프 및 검증 호출
├── test.py                평가 진입점, 모델 로드 후 eval_lane() 호출
├── demo.py                추론 결과를 AVI 영상으로 저장
├── speed_simple.py        FPS 측정
│
├── configs/               데이터셋 × 백본 조합별 하이퍼파라미터
│   ├── tusimple_res18.py
│   ├── tusimple_res34.py
│   ├── culane_res18.py
│   ├── culane_res34.py
│   ├── curvelanes_res18.py
│   └── curvelanes_res34.py
│
├── model/
│   ├── backbone.py        ResNet/VGG 백본, 3단계 feature 반환
│   ├── model_culane.py    parsingNet (CULane/TuSimple용 핵심 모델)
│   ├── model_tusimple.py  parsingNet 래퍼 (내부적으로 model_culane 사용)
│   ├── model_curvelanes.py CurveLanes 전용 모델 (lane token 추가)
│   ├── seg_model.py       보조 세그멘테이션 헤드
│   └── layer.py           CoordConv 등 커스텀 레이어
│
├── data/
│   ├── dataset.py         LaneDataset, LaneTestDataset (이미지 로딩)
│   ├── dataloader.py      테스트용 DataLoader
│   ├── dali_data.py       NVIDIA DALI 기반 고속 학습 데이터 로더
│   ├── mytransforms.py    데이터 증강 (랜덤 크롭/플립/색상 변환)
│   └── constant.py        row/col anchor 상수 정의
│
├── utils/
│   ├── common.py          모델/데이터로더 생성, 추론, 손실 계산, 저장
│   ├── config.py          addict 기반 config 파서
│   ├── loss.py            SoftmaxFocalLoss, MeanLoss, VarLoss 등
│   ├── factory.py         손실 딕셔너리, 옵티마이저, 스케줄러 팩토리
│   ├── metrics.py         AccTopk, MultiLabelAcc, Metric_mIoU 등
│   └── dist_utils.py      분산 학습 유틸, TensorBoard 래퍼
│
├── evaluation/
│   ├── eval_wrapper.py    데이터셋별 평가 진입점 (eval_lane 함수)
│   ├── tusimple/          TuSimple 공식 평가 스크립트
│   └── culane/            CULane C++ 평가 바이너리 소스
│
├── scripts/               데이터셋 전처리 스크립트
│   ├── convert_tusimple.py
│   ├── convert_curvelanes.py
│   ├── cache_culane_ponits.py
│   └── make_curvelane_as_culane_test.py
│
├── deploy/
│   ├── pt2onnx.py         PyTorch → ONNX 변환
│   └── trt_infer.py       TensorRT 추론
│
└── my_interp/             커스텀 CUDA 보간 확장 모듈
    ├── my_interp_cuda.cpp
    ├── my_interp_cuda_kernel.cu
    └── setup.py
```
