# Trainer와 미니배치

[API 레퍼런스](api.md) · [빠른 시작](quickstart.md) · [실습 예제](examples.md)

[English](../en/trainer.md) · [日本語](../ja/trainer.md) · [한국어](../ko/trainer.md)

## 설정

`TensorRuby::Training::Trainer.new(model, optimizer:, loss:, weight: nil, max_grad_norm: nil, forward: nil)`

- `optimizer:`는 `zero_grad`와 `step`을 제공합니다. `loss:`는 예측과 정답을 받아 유한한 스칼라 `Value`를 반환해야 합니다.
- `weight:`의 기본값은 입력·정답 쌍당 1입니다. 원소 평균 손실은 정답 원소 수로, 마스크를 쓰는 언어 모델 손실은 학습 대상 토큰 수로 가중치를 줄 수 있습니다. 가중치가 0이면 건너뜁니다.
- `max_grad_norm:`은 업데이트 전 전체 기울기 노름을 제한합니다. `forward:`에는 기본 `model.call(input)` 대신 실행할 함수를 전달할 수 있습니다.

## 학습과 평가

| API | 입력과 반환값 |
| --- | --- |
| `train_epoch(examples, accumulation_steps: 1, learning_rate: nil)` | 학습 모드에서 가중 평균 손실을 반환합니다. 최대 `accumulation_steps`개 쌍마다 한 번 업데이트하며, 마지막에 남은 쌍도 처리합니다. |
| `evaluate(examples)` | 기울기를 기록하지 않고 가중 평균 손실을 반환합니다. 평가가 끝나면 이전 학습·평가 모드로 돌아갑니다. |
| `steps` | 이전 에포크를 포함해 완료된 옵티마이저 업데이트 횟수를 반환합니다. |

`learning_rate:` 함수는 0부터 시작하는 업데이트 번호를 받아 학습률을 반환합니다. 옵티마이저가 `step(lr:)`을 지원해야 합니다. `Optim::Adam`이 이를 지원합니다.

## 실행 예제

로더는 샘플 두 개와 한 개를 각각 배치로 반환합니다. 정답 원소 수로 가중하면 모든 관측값을 같은 비중으로 다루며, 두 배치를 누적하면 옵티마이저가 한 번 업데이트됩니다.

```ruby
require 'tensorruby'

samples = [[[0], [1]], [[1], [3]], [[2], [5]]]
loader = TensorRuby::Data::DataLoader.new(samples, batch_size: 2, shuffle: false)
model = TensorRuby::NN::Linear.new(1, 1, random: Random.new(42))
optimizer = TensorRuby::Optim::SGD.new(model.parameters, lr: 0.05)
trainer = TensorRuby::Training::Trainer.new(
  model, optimizer: optimizer, loss: TensorRuby::NN::MSELoss.new,
  weight: ->(target) { target.size }
)

trainer.train_epoch(loader, accumulation_steps: 2)
p trainer.steps                 # 1
300.times { trainer.train_epoch(loader, accumulation_steps: 2) }
p trainer.evaluate(loader)      # approximately 0.0
```

## 오류 조건

누적 횟수가 양수가 아니거나, 가중치가 음수 또는 유한하지 않거나, 손실이 유한하지 않거나, 양의 가중치가 있는 샘플이 없으면 `ArgumentError`가 발생합니다. `no_grad` 안에서 학습할 때도 같은 오류가 발생합니다.
