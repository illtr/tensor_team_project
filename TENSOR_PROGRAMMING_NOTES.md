# Tensor Programming 강의 정리 (Lec 4~8)

> DAI2001 - Kwon, Bokyung 교수님 강의 내용 정리

---

## Lec 4 : Tensor Operation

### 최소값 / 최대값 연산

| 함수 | 반환값 |
|---|---|
| `torch.min(tensor)` | 전체 최소값 tensor |
| `torch.min(tensor, dim)` | `values`, `indices` |
| `torch.min(tensor, dim, keepdim=True)` | 형태 유지된 `values`, `indices` |
| `torch.max(tensor)` | 전체 최대값 tensor |
| `torch.argmin(tensor)` | 최소값 위치 tensor |
| `torch.argmax(tensor)` | 최대값 위치 tensor |

```python
values, indices = x.min(dim=0)
values, indices = x.max(dim=0, keepdim=True)
x.argmin()          # 전체 최소 위치
x.argmin(dim=0)     # 열 기준 최소 위치
```

### 합과 곱

```python
torch.sum(x)           # 전체 합
torch.sum(x, dim=0)    # 열 기준 합
torch.sum(x, dim=1)    # 행 기준 합
torch.prod(x)          # 전체 곱
```

### 기초 통계량

```python
torch.mean(x)                    # 평균
torch.var(x)                     # 분산 (불편 추정량)
torch.std(x)                     # 표준편차
variance, mean = torch.var_mean(x)
std, mean      = torch.std_mean(x)
```

### 인덱스 찾기

```python
# condition이 True면 x, False면 y 반환
z = torch.where(x > 0, x, y)
```

### 정렬

```python
sorted_values, indices = torch.sort(x)          # 오름차순 정렬
torch.argsort(x)                                 # 정렬 인덱스만 반환
torch.sort(x, dim=0)                             # 열 기준 정렬
```

### 행렬 연산

```python
C = A.matmul(B)         # 내적 (행렬 곱)
C = A @ B               # 동일한 결과
C = A.mm(B)             # 2D 행렬 곱
C = A.bmm(B)            # 배치 행렬 곱

# 선형대수
from torch import linalg as LA
LA.norm(A)      # 노름
LA.det(A)       # 행렬식
A.T             # 전치 (= torch.transpose())
LA.inv(A)       # 역행렬
LA.matrix_rank(A)           # rank
LA.solve(A, b)              # 선형방정식 Ax = b 의 해

torch.allclose(A, B)        # 두 텐서의 값이 같은지 확인
```

---

## Lec 5 : Autograd

### 자동 미분 개요

- `torch.autograd`로 경사하강법에 필요한 미분값을 자동 계산
- **Gradient vector**: x 위치에서 f(x)의 변화가 가장 큰 방향과 크기
- **Computational graph**: 연쇄 법칙을 사용하여 미분을 자동 계산

### tensor.backward()

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.retain_grad()   # 중간 노드의 미분값 유지
y.backward()

print(x.grad)     # dy/dx = 2x = 4.0
print(y.grad)     # dy/dy = 1.0

x.grad.zero_()    # grad를 0으로 초기화 (재사용 전 필수)
```

### 계산 그래프

```
x ──→ * ──→ + ──→ z ──→ CE ──→ loss
      ↑       ↑
      w       b         y
   (Parameters)      (정답)
```

- `requires_grad=True`: 계산 그래프에 자동 미분 허용
- `detach()`: 텐서를 계산 그래프에서 분리
- `retain_graph=True`: 중간 노드를 재사용 가능하게 유지

### 주요 미분 예시

```python
# 스칼라
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
# x.grad = 2x = 4.0

# 벡터
x = torch.tensor([2., 3., 4.], requires_grad=True)
y = x ** 2
y.backward(gradient=torch.ones_like(y))
# x.grad = [4., 6., 8.]

# 다변수 편미분
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x**2 + y**2
z.backward()
# x.grad = 2x = 4.0,  y.grad = 2y = 6.0
```

### autograd.grad()

```python
dz_dx = torch.autograd.grad(outputs=z, inputs=x)
dz_dy = torch.autograd.grad(outputs=z, inputs=y, retain_graph=True)

# 여러 출력의 gradient 합계 계산
dzw_dxy = torch.autograd.grad(outputs=[z, w], inputs=[x, y],
                                grad_outputs=[v, v])
# dzw_dxy[0] = dz/dx + dw/dx
# dzw_dxy[1] = dz/dy + dw/dy
```

### Jacobian 계산

```python
from torch.autograd.functional import jacobian

J = jacobian(f, (x1, x2, x3))
```

---

## Lec 6 : Gradient Descent and Optimization

### 경사하강법

$$x_{t+1} = x_t - \eta \cdot \nabla f(x_t)$$

- $\eta$: 학습률 (learning rate)
- $\nabla f(x_t)$: 1차 편미분값 (gradient)
- 초기값에 따라 국소 최솟값에 빠질 수 있음

```python
# 수치 미분 (중심 차분)
def diff(f, x):
    h = 0.001
    return (f(x+h) - f(x-h)) / (2*h)

# 경사하강법 구현
while True:
    y = f(x)
    y.backward()
    x.data = x.data - lr * x.grad
    x.grad.zero_()
```

### torch.optim 최적화 알고리즘

#### ① SGD (Stochastic Gradient Descent, 1950~)

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla f(\theta_t; x_i, y_i)$$

- 전체 데이터 중 1개 (또는 mini-batch)를 랜덤 선택해 gradient 계산
- oscillation, saddle point, ravine 구조에서 수렴 어려움

**Polyak's Momentum**:
$$v_{t+1} = \mu \cdot v_t - \eta \cdot \nabla f(\theta_t)$$
$$\theta_{t+1} = \theta_t + v_{t+1}$$

**NAG (Nesterov's Accelerated Gradient, 2013)**:
$$v_{t+1} = \mu \cdot v_t - \eta \cdot \nabla f(\theta_t + \mu \cdot v_t)$$

#### ② AdaGrad (Adaptive Gradient, 2011)

$$G_t = G_{t-1} + (\nabla f(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \varepsilon}} \cdot \nabla f(\theta_t)$$

- 파라미터별 개별 학습률 적용
- $G_t$가 단조 증가 → 유효학습률이 0으로 수렴해 조기 종료 가능

#### ③ RMSprop (2012)

$$E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho)(\nabla f(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \varepsilon}} \cdot \nabla f(\theta_t)$$

- AdaGrad의 학습률 급감 문제를 지수 이동 평균으로 개선
- $\rho = 0.9$ 사용 (최근 gradient에 높은 가중치)

#### ④ Adam (Adaptive Momentum Estimation, 2015)

$$m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot \nabla f(\theta_t)$$
$$v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot (\nabla f(\theta_t))^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

- 1차 모멘트(Polyak's momentum) + 2차 모멘트(RMSprop) 결합
- $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$ 기본값
- 대부분의 딥러닝에서 안정적이고 빠르게 수렴

#### ⑤ AdamW (Adam with decoupled Weight decay, 2019)

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} - \eta\lambda\theta_t$$

- Adam에서 weight decay를 gradient와 분리
- 모든 파라미터에 동일하게 weight decay 적용
- Transformer / ViT / BERT 학습에 권장

```python
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(params, lr=0.001)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

#### 비교 요약

| 알고리즘 | 특징 |
|---|---|
| SGD | 기본 경사하강법 + 모멘텀 |
| AdaGrad | 파라미터별 과거 gradient 제곱합 적용 |
| RMSprop | AdaGrad의 학습률 급감 개선 |
| Adam | AdaGrad + RMSprop 조합 |
| AdamW | Transformer 계열 딥러닝에 적합 |

---

## Lec 7 : nn.Module, 활성화 함수, 손실 함수

### Mini-batch와 경사하강법

- **Mini-batch**: 일정 개수를 임의로 추출한 단위
- 국소 최솟값 회피 + 대용량 데이터에서 유리
- `torch.utils.data`의 `Dataset`, `DataLoader` 활용

```python
# DataLoader 사용
dataset = TensorDataset(x, t)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### nn.Module 상속 구조

```python
class LinearModel(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
model.parameters()           # 파라미터 확인
model.state_dict()           # 레이어명 → 파라미터 딕셔너리

torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

- `model(x)` 호출 시 `forward(x)` 자동 실행
- `state_dict()`: `collections.OrderedDict` 형태 반환

### 활성화 함수

| 함수 | 수식 | 범위 |
|---|---|---|
| `nn.Sigmoid()` | $\frac{1}{1+e^{-x}}$ | (0, 1) |
| `nn.Tanh()` | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1) |
| `nn.ReLU()` | $\max(0, x)$ | $[0, \infty)$ |
| `nn.LeakyReLU()` | $x$ if $x \geq 0$ else $\alpha x$ | $(-\infty, \infty)$ |
| `nn.Softmax(dim)` | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | (0, 1), 합=1 |
| `nn.LogSigmoid()` | $-\log(1+e^{-x})$ | $(-\infty, 0)$ |

- **개별 적용**: Sigmoid, Tanh, ReLU, LeakyReLU
- **층 전체 적용**: Softmax, LogSoftmax

```python
f = nn.Sigmoid()
f = nn.ReLU()
f = nn.LeakyReLU()
f = nn.Tanh()
f = nn.Softmax(dim=1)
```

### 손실 함수

| 함수 | 용도 | 수식 |
|---|---|---|
| `nn.MSELoss()` | 회귀 | $\ell_n = (x_n - y_n)^2$ |
| `nn.BCELoss()` | 이진 분류 | $\ell_n = -w_n[y_n \log x_n + (1-y_n)\log(1-x_n)]$ |
| `nn.CrossEntropyLoss()` | 다중 분류 | Softmax + NLLLoss |
| `nn.NLLLoss()` | 음의 로그우도 | $\ell_n = -w_{y_n} x_{n,y_n}$ |

```python
loss_fn = nn.MSELoss()
loss_fn = nn.BCELoss(reduction='mean')
loss_fn = nn.CrossEntropyLoss(reduction='mean')

loss = loss_fn(y_pred, y_true)
```

---

## Lec 8 : Initialization, Normalization, Dropout

### 가중치 초기화

초기값은 최적화 결과에 영향을 미치며, 잘못된 초기화는 gradient vanishing/exploding을 유발한다.

#### Xavier 초기화 (Glorot and Bengio, 2010)

- 층 간 gradient를 비슷하게 유지, activation 분산 일정하게 유지
- **Uniform**: $W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}\right]$
- **Normal**: $W \sim \mathcal{N}\left(0, \frac{2}{n_{in}+n_{out}}\right)$
- linear, tanh, sigmoid 활성화 함수에 적합

```python
nn.init.xavier_normal_(m.weight)
nn.init.xavier_uniform_(m.weight)
```

#### Kaiming He 초기화 (He et al, 2015)

- ReLU / LeakyReLU 활성화 함수에 적합
- $\text{Var}(W) = \frac{2}{(1+a^2) \cdot n_{in}}$, $\text{std} = \frac{\text{gain}}{\sqrt{n_{in}}}$

```python
nn.init.kaiming_normal_(m.weight)
nn.init.kaiming_uniform_(m.weight)
```

#### 모델에 적용하는 패턴

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

model.apply(init_weights)
```

### 가중치 규제 (Regularization)

- L1: 가중치 절대값의 합 → 희소성 유도
- L2: 가중치 제곱합 → optimizer의 `weight_decay` 파라미터로 적용

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

- `weight_decay`가 너무 작으면 영향 없음, 너무 크면 최적화 성능 저하

### 배치 정규화 (Batch Normalization)

$$y = \frac{x - E(x)}{\sqrt{Var(x) + \varepsilon}}$$

- 활성화 함수 **이전**에 적용
- 가중치 초기화 영향을 줄이고 과적합 방지
- mini-batch 단위로 층 출력을 평균 0, 분산 1로 정규화

```python
# 1D (FC layer 후)
BN = nn.BatchNorm1d(n_feature)
out = BN(y)

# 2D (Conv layer 후)
BN = nn.BatchNorm2d(C)
out = BN(y)
```

- 학습 후 `out_mean ≈ 0`, `out_std ≈ 1` 확인 가능

### Dropout

- 학습 중 p 확률로 노드 값을 0으로 만들어 과적합 방지
- **평가(eval) 시에는 모든 노드 사용** (비활성화 안 됨)

```python
model.train()                    # dropout 활성화
out = nn.Dropout(0.4)(y)

model.eval()                     # dropout 비활성화
with torch.no_grad():
    out = model(x_test)
```

```python
# inplace=True: 입력 텐서를 직접 수정
out = nn.Dropout(0.4, inplace=True)(y)
```

- dropout 후 살아남은 값은 $\frac{1}{1-p}$로 스케일 조정됨 (기댓값 보존)

---

## 전체 학습 파이프라인 요약

```python
# 1. 데이터 준비 (Lec 7)
dataset = TensorDataset(x_train, y_train)
loader  = DataLoader(dataset, batch_size=8, shuffle=True)

# 2. 모델 정의 (Lec 7)
class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.bn  = nn.BatchNorm1d(64)      # Lec 8
        self.drop = nn.Dropout(0.3)        # Lec 8
        self.fc2 = nn.Linear(64, 3)
    def forward(self, x):
        x = self.drop(self.bn(torch.relu(self.fc1(x))))
        return self.fc2(x)

model = MyMLP()
model.apply(init_weights)                  # Lec 8 초기화

# 3. 손실함수 & 옵티마이저 (Lec 6, 7)
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 4. 훈련 루프 (Lec 5, 6)
for epoch in range(100):
    model.train()
    for x, y in loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()    # grad 초기화
        loss.backward()          # 역전파 (autograd)
        optimizer.step()         # 파라미터 갱신

# 5. 평가 (Lec 7, 8)
model.eval()
with torch.no_grad():
    out  = model(x_test)
    pred = out.argmax(dim=1)
    acc  = (pred == y_test).float().mean()
```
