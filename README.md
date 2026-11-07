# rubygrad

Pure Ruby로 작성한 작은 딥러닝 프레임워크입니다. 외부 딥러닝 프레임워크나 네이티브 수치 라이브러리에 의존하지 않습니다. Ruby 2.7 이상이 필요합니다.

## 실행

```sh
ruby -Ilib test/rubygrad_test.rb
ruby examples/regression.rb
gem build rubygrad.gemspec
```

`rake`가 설치되어 있다면 `rake test`도 사용할 수 있습니다. 런타임 의존성은 없으며 테스트에는 Minitest가 필요합니다.

## 사용법

```ruby
require 'rubygrad'
x = Rubygrad::Tensor.new([[1, 2], [3, 4]])
w = Rubygrad::Tensor.new([[2], [1]])
output = x.matmul(w) + 1
output.sum.backward
p output.to_a            # [[5.0], [11.0]]
p w.values.map(&:grad)  # [4.0, 6.0]
```

`Value#backward(seed = 1.0)`는 그래프를 위상 정렬해 역전파합니다. 공유 노드의 기울기를 합산하며, 재호출할 때 중간 노드 기울기는 초기화하고 리프 기울기는 누적합니다. 새 학습 단계 전에는 `optimizer.zero_grad`를 호출하세요. 역전파는 순전파 당시의 로컬 미분을 사용하므로 매개변수 갱신 후에는 새로운 순전파 그래프를 만들어야 합니다.

`Tensor`는 연속된 row-major 저장소와 shape/stride를 가지며 각 원소가 `Value`입니다. 원소별 `+`, `-`, `*`, `/`는 뒤쪽 축 기준 broadcasting을 지원합니다. `matmul`은 현재 2차원 행렬만 지원하며, `sum`과 `mean`은 전체 원소를 축소해 `Value`를 반환합니다. 빈 차원, 텐서 뷰, 배치 matmul, 축별 축소, dtype 및 장치 선택은 아직 지원하지 않습니다. `Linear` 입력은 `[batch, features]` 형태입니다. MSE는 예측과 정답의 shape가 정확히 일치해야 합니다.

## 확장 방향

현재 텐서 구현은 정확성 검증을 위한 스칼라 그래프 방식이므로 큰 모델에는 느리고 메모리 사용량이 큽니다. 향후 shape/stride 메타데이터와 연산 API를 유지하면서 저장소와 커널을 백엔드 인터페이스로 분리하고, 텐서 단위 VJP를 도입해 Rust/C++ CPU 또는 GPU 백엔드를 연결할 수 있습니다. 현재 이러한 백엔드나 연결 코드는 포함하지 않습니다.
