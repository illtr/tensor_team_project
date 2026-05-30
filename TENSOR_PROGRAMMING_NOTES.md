# 강의 내용 → 코드 대응 정리

> DAI2001 Tensor Programming (Lec 4~8) 내용이 이 프로젝트 코드에 어떻게 반영되어 있는지 정리

---

## Lec 4 : Tensor Operation

**반영 위치**: `utils/loss.py`, `model/model_culane.py`

- `torch.arange()`, `.view()`, `.softmax()`, `.sum()` 등 텐서 연산으로 soft argmax 구현
- `torch.cat()`, `.reshape()` 으로 예측 출력 텐서 조합
- `torch.nn.functional.one_hot()` 으로 라벨 원핫 인코딩

```python
# utils/loss.py
grid = torch.arange(c, device=logits.device).view(1, c, 1, 1)
logits = (logits.softmax(1) * grid).sum(1)   # soft argmax
```

---

## Lec 5 : Autograd

**반영 위치**: `train.py`

- `loss.backward()` 로 역전파 수행
- `optimizer.zero_grad()` 로 gradient 초기화 후 재사용

```python
# train.py
loss = calc_loss(loss_dict, results, logger, global_step, epoch)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Lec 6 : Gradient Descent and Optimization

**반영 위치**: `utils/factory.py`

- `torch.optim.Adam`, `torch.optim.SGD` 선택적 사용
- `weight_decay` 로 L2 규제 적용
- `CosineAnnealingLR`, `MultiStepLR` 학습률 스케줄러 직접 구현

```python
# utils/factory.py
optimizer = torch.optim.Adam(training_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
optimizer = torch.optim.SGD(training_params, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
```

---

## Lec 7 : nn.Module, 활성화 함수, 손실 함수

**반영 위치**: `model/model_culane.py`, `utils/loss.py`, `utils/factory.py`

- `parsingNet`이 `torch.nn.Module` 상속, `super().__init__()` 호출, `forward()` 구현
- `nn.Linear`, `nn.ReLU`, `nn.Sequential` 로 분류 헤드 구성
- `nn.CrossEntropyLoss`, `nn.BCELoss`, `SoftmaxFocalLoss` 등 다양한 손실 함수 사용
- `TensorDataset`, `DataLoader` 로 mini-batch 학습 구성

```python
# model/model_culane.py
class parsingNet(torch.nn.Module):
    def __init__(self, ...):
        super(parsingNet, self).__init__()
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, mlp_mid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_mid_dim, self.total_dim),
        )
    def forward(self, x): ...
```

---

## Lec 8 : Initialization, Normalization, Dropout

**반영 위치**: `utils/common.py`, `model/model_culane.py`

- `initialize_weights()` 함수로 Xavier 초기화 적용
- `fc_norm=True` 옵션 시 `nn.LayerNorm` 삽입, 아니면 `nn.Identity()` 로 skip
- ResNet backbone 내부에 BatchNorm 사용

```python
# model/model_culane.py
torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity()

# utils/common.py
initialize_weights(self.cls)   # Xavier 초기화
```
