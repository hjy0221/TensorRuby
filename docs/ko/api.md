# API 레퍼런스

[홈](../../README.ko.md) · [빠른 시작](quickstart.md) · [API 레퍼런스](api.md) · [실습 예제](examples.md) · [개발 문서](development.md)

[English](../en/api.md) · [日本語](../ja/api.md) · [한국어](../ko/api.md)

별도 표시가 없으면 아래 이름은 `TensorRuby` 네임스페이스에 속합니다.

## 텐서와 자동미분

[텐서와 자동미분](tensor.md) — 입력 형태, 연산 시그니처, 시드, 기울기 누적.

## 신경망

- `NN::Module#parameters`는 학습 가능한 스칼라를 모읍니다. `train`과 `eval`은 하위 계층에도 적용됩니다.
- `NN::Sequential.new(*layers)`는 층을 순서대로 실행합니다. `NN::Linear.new(in_features, out_features)`는 마지막 축이 특성인 2차원 이상 입력을 받습니다.
- `NN::ReLU`, `NN::GELU`, `NN::LayerNorm.new(features)`, `NN::Dropout.new(p = 0.5)`를 사용할 수 있습니다. LayerNorm은 행렬을 받고, Dropout은 학습 모드에서 동작합니다.
- `NN::MSELoss#call(prediction, target)`은 형태가 같은 입력을 받습니다. `NN::CrossEntropyLoss.new(ignore_index: nil)`는 `[batch, classes]` 로짓과 `[batch]` 클래스 인덱스를 받습니다.

## 학습과 데이터

- `Data::DataLoader.new(dataset, batch_size:, shuffle: true, random: Random.new)`는 인덱스로 접근할 수 있는 데이터셋에서 텐서 쌍을 반환합니다. 마지막에 남은 배치도 포함합니다.
- `Optim::SGD.new(parameters, lr:)`와 `Optim::Adam.new(parameters, lr: 0.001)`은 `zero_grad`와 `step`을 제공합니다. Adam은 `step(lr:)`도 지원합니다.
- [Trainer와 미니배치](trainer.md) — 가중 손실, 마지막 미니배치, 기울기 누적, 평가.

## 텍스트와 생성

- `NN::SelfAttention`과 `NN::TransformerBlock`은 `[sequence, features]` 행렬 한 개를 받습니다. `NN::GPT.new(vocab_size:, context_size:, features:, heads:, layers:)`는 정수 토큰 시퀀스 한 개에서 `[sequence, vocabulary]` 로짓을 반환합니다.
- `GPT#generate(ids, max_new_tokens:, stop_token: nil)`은 매 단계에서 가장 높은 점수의 토큰을 선택하며, 입력과 생성 토큰을 함께 반환합니다.
- `Text::BytePairTokenizer.train(documents, max_merges: 64)`는 UTF-8 바이트 병합 규칙을 학습합니다. `Text::NextTokenWindows`는 입력과 한 토큰씩 밀린 정답을 반환합니다.
- `Text::Checkpoint.save(path, model, vocabulary)`는 설정, 바이트 병합 규칙, 가중치를 저장합니다. `load(path)`로 모델과 바이트 어휘를 복원합니다. `Chat::Session#reply`는 대화를 토큰화하고 답변을 텍스트로 반환합니다.

## 확산 모델

- `Diffusion::Schedule.new(steps:, beta_start: 0.0001, beta_end: 0.02)`는 선형 DDPM 일정을 만듭니다.
- `add_noise(clean, noise, step)`의 단계는 0부터 시작합니다. `sample(shape, on_step: nil) { |noisy, step| … }` 블록은 같은 형태의 노이즈 예측을 반환해야 합니다. 콜백은 남은 단계 수와 현재 샘플을 받습니다.
