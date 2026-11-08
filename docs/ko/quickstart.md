# 빠른 시작

[홈](../../README.ko.md) · [빠른 시작](quickstart.md) · [API 레퍼런스](api.md) · [실습 예제](examples.md) · [개발 문서](development.md)

[English](../en/quickstart.md) · [日本語](../ja/quickstart.md) · [한국어](../ko/quickstart.md)

터미널에서 아래 명령을 실행하세요. gem에 예제가 포함되어 있어 어느 폴더에서든 실행할 수 있습니다.

```sh
curl -fL https://github.com/hjy0221/TensorRuby/releases/latest/download/tensorruby.gem -o tensorruby.gem
gem install --local tensorruby.gem
tensorruby run xor
```

## 첫 모델 학습

아래 코드를 `train.rb`로 저장하고 `ruby train.rb`로 실행하세요. 모델은 `y = 2x + 1`을 학습합니다. Trainer에 넘기는 각 쌍은 입력 배치와 정답 배치입니다.

```ruby
require 'tensorruby'

x = TensorRuby.tensor([[0], [1], [2]])
y = TensorRuby.tensor([[1], [3], [5]])
model = TensorRuby::NN::Linear.new(1, 1, random: Random.new(42))
optimizer = TensorRuby::Optim::SGD.new(model.parameters, lr: 0.05)
trainer = TensorRuby::Training::Trainer.new(
  model, optimizer: optimizer, loss: TensorRuby::NN::MSELoss.new
)

data = [[x, y]]
200.times { trainer.train_epoch(data) }
p trainer.evaluate(data)
model.eval
p TensorRuby.no_grad { model.call([[3]]).to_a } # approximately [[7.0]]
```

## 다음 단계

`tensorruby examples`로 전체 예제를 확인하세요. 명령을 찾을 수 없다면 `gem environment`에서 gem 실행 파일 폴더를 확인해 PATH에 추가하세요.
