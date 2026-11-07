# TensorRuby

[English](README.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

Pure Ruby로 작성한 작은 딥러닝 프레임워크입니다. 외부 딥러닝 프레임워크나 네이티브 수치 라이브러리에 의존하지 않습니다. Ruby 2.6 이상이 필요합니다.

## 설치

GitHub 최신 릴리스에서 gem을 설치합니다.

```sh
curl -fL https://github.com/hjy0221/TensorRuby/releases/latest/download/tensorruby.gem -o tensorruby.gem
gem install --local tensorruby.gem
```

## 실행

```sh
ruby -Ilib test/tensorruby_test.rb
ruby examples/regression.rb
gem build tensorruby.gemspec
```

`rake`가 설치되어 있다면 `rake test`도 사용할 수 있습니다. 런타임 의존성은 없으며 테스트에는 Minitest가 필요합니다.

## 사용법

설치한 뒤 아래 코드를 `ruby`로 실행하세요.

```ruby
require 'tensorruby'
x = TensorRuby::Tensor.new([[1, 2], [3, 4]])
w = TensorRuby::Tensor.new([[2], [1]])
output = x.matmul(w) + 1
output.sum.backward
p output.to_a            # [[5.0], [11.0]]
p w.values.map(&:grad)  # [4.0, 6.0]
```

`Value#backward(seed = 1.0)`는 그래프를 위상 정렬해 역전파합니다. 공유 노드의 기울기를 합산하며, 재호출할 때 중간 노드 기울기는 초기화하고 리프 기울기는 누적합니다. 새 학습 단계 전에는 `optimizer.zero_grad`를 호출하세요. 역전파는 순전파 당시의 로컬 미분을 사용하므로 매개변수 갱신 후에는 새로운 순전파 그래프를 만들어야 합니다.

`Tensor`는 연속된 row-major 저장소와 shape/stride를 가집니다. 인덱싱과 `values`로 스칼라 `Value`에 접근할 수 있습니다. 원소별 `+`, `-`, `*`, `/`는 뒤쪽 축 기준 broadcasting을 지원합니다. `matmul`은 현재 2차원 행렬만 지원하며, `sum`과 `mean`은 전체 원소를 축소해 `Value`를 반환합니다. 빈 차원, 텐서 뷰, 배치 matmul, 축별 축소, dtype 및 장치 선택은 아직 지원하지 않습니다. `Linear` 입력은 `[batch, features]` 형태입니다. MSE는 예측과 정답의 shape가 정확히 일치해야 합니다.

## XOR 모델 학습

`TensorRuby.tensor`로 텐서를 만들고 `NN::Sequential`로 층을 연결합니다. 전체 예제는 `ruby examples/xor.rb`로 실행할 수 있습니다.

```ruby
require 'tensorruby'

nn = TensorRuby::NN
random = Random.new(42)
x = TensorRuby.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
target = TensorRuby.tensor([[0], [1], [1], [0]])

model = nn::Sequential.new(
  nn::Linear.new(2, 8, random: random),
  nn::ReLU.new,
  nn::Linear.new(8, 1, random: random)
)
criterion = nn::MSELoss.new
optimizer = TensorRuby::Optim::SGD.new(model.parameters, lr: 0.1)

500.times do
  optimizer.zero_grad
  loss = criterion.call(model.call(x), target)
  loss.backward
  optimizer.step
end
```

## 다중 클래스 분류

`NN::CrossEntropyLoss`에는 `[batch, classes]` 형태의 로짓과 `[batch]` 형태의 정수 클래스 인덱스를 전달합니다. `Optim::Adam`은 학습 단계 사이에 모멘트를 유지합니다. `ruby examples/classification.rb`로 3개 클래스를 분류하는 예제를 실행할 수 있습니다.

`Data::DataLoader.new(samples, batch_size: 8)`은 `[입력, 정답]` 쌍을 미니배치로 묶고 에포크마다 섞습니다.

## 학습과 평가

`model.train`은 드롭아웃을 켜고, `model.eval`은 하위 층까지 드롭아웃을 끕니다. 추론을 `TensorRuby.no_grad`로 감싸면 새 연산을 역전파 그래프에 기록하지 않습니다.

```ruby
model.eval
predictions = TensorRuby.no_grad { model.call(x).to_a }
```

## GPT와 확산 모델 예제

GPT-2 형태의 디코더는 정수 토큰 ID를 받아 다음 토큰을 예측하도록 학습합니다. 학습 가능한 토큰·위치 임베딩, causal 멀티헤드 어텐션, 사전 레이어 정규화, GELU, 임베딩과 출력 가중치 공유를 지원합니다. 확산 예제는 합성 점과 흑백 줄무늬를 학습하며, 이미지 예제는 생성한 6×6 이미지 8장을 PNG로 저장합니다.

```sh
ruby examples/gpt.rb
ruby examples/transformer.rb
ruby examples/diffusion.rb
ruby examples/image_diffusion.rb diffusion-samples.png diffusion-trajectory.png
```

## 대화 모델과 몬스터 스프라이트

기본 한국어·일본어·영어 대화로 작은 모델을 학습하거나 `--data FILE`에 `[사용자, 어시스턴트]` 쌍으로 이루어진 대화 배열을 JSON으로 전달할 수 있습니다. 공백으로 나눈 학습 어휘를 사용합니다. 스프라이트는 고정 실루엣에 확산 모델이 생성한 색과 눈 색 변형을 적용합니다.

```sh
ruby examples/chat.rb --interactive
ruby examples/monster_diffusion.rb all monster-sprites.png
```

## 다국어 수도 모델 학습과 불러오기

12개 나라의 수도를 세 언어로 처음부터 학습하고, 저장한 가중치를 재학습 없이 불러올 수 있습니다. 학습에 쓴 질문과 같은 단어를 재배열한 질문의 정답률을 각각 표시합니다.

```sh
ruby examples/capital_chat.rb --save capital-chat-weights.json
ruby examples/capital_chat.rb --load capital-chat-weights.json --interactive
```

## 원문으로 사전학습

바이트 페어 토크나이저와 작은 GPT를 처음부터 학습합니다. 유효한 UTF-8 텍스트를 토큰화할 수 있습니다. 한 줄에 문서 하나를 담은 UTF-8 학습·검증 파일을 각각 전달하세요. 토큰 병합 규칙은 학습 텍스트에서만 배웁니다.

```sh
ruby examples/pretrain.rb
ruby examples/pretrain.rb --train train.txt --validation validation.txt --epochs 12
```

## 체크포인트 미세조정과 텍스트 생성

사전학습은 여러 시퀀스의 기울기를 토큰 수에 비례해 누적하고 전체 기울기 노름을 제한합니다. 학습률은 워밍업 뒤 코사인 형태로 줄입니다. 설정·가중치·병합 규칙은 JSON으로 저장하며, 미세조정의 손실에는 어시스턴트 토큰만 포함합니다.

```sh
ruby examples/pretrain.rb --save pretrained.json
ruby examples/finetune.rb --save chat-model.json pretrained.json
ruby examples/generate.rb --chat chat-model.json "하늘?"
```

## 확장 방향

텐서 연산 하나마다 그래프 노드 하나를 기록하고 벡터–야코비안 곱으로 역전파합니다. `output.backward(seed)`의 시드는 출력과 형태가 같아야 하며, 원소가 하나면 생략할 수 있습니다. 현재 CPU 커널은 Ruby로 실행되며, 향후 네이티브 구현으로 교체할 수 있습니다.
