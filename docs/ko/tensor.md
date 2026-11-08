# 텐서와 자동미분

[API 레퍼런스](api.md) · [빠른 시작](quickstart.md) · [실습 예제](examples.md)

[English](../en/tensor.md) · [日本語](../ja/tensor.md) · [한국어](../ko/tensor.md)

## 생성과 구조 확인

- `TensorRuby.tensor(data, shape: nil)`은 수치 스칼라나 직사각형 중첩 배열로 텐서를 만듭니다. `shape:`를 지정하면 각 차원의 곱이 값의 개수와 일치해야 합니다.
- `shape`는 각 차원의 크기, `strides`는 행 우선 저장소에서의 원소 간격, `rank`는 축 수, `size`는 원소 수입니다.
- `to_a`는 수치의 중첩 배열을 반환하며 0차원에서는 수치 하나를 반환합니다. `tensor[i, j]`와 `values`로 스칼라 `Value`에 접근할 수 있습니다.

## 연산과 입력 형태

| API | 입력과 반환값 |
| --- | --- |
| `+`, `-`, `*`, `/` | 원소별 연산은 뒤쪽 축부터 맞추고 크기가 1인 축을 확장합니다. |
| `matmul(other)` | `[..., M, K]`와 `[..., K, N]`을 곱하면 `[..., M, N]`이 됩니다. 앞쪽 축은 브로드캐스팅할 수 있어야 하고 양쪽 모두 2차원 이상이어야 합니다. |
| `sum`, `mean` | 전체 원소의 합이나 평균을 스칼라 `Value`로 반환합니다. |
| `transpose` | `transpose`는 2차원 텐서를 받아 `[N, M]` 형태의 새 텐서를 반환합니다. |
| `softmax(causal: false)`, `normalize_rows(eps: 1e-5)` | 2차원 텐서를 받아 같은 형태로 반환합니다. Causal softmax는 현재 행 번호보다 뒤에 있는 열을 0으로 만듭니다. |

## 역전파

`Tensor#backward(seed)`는 출력과 형태가 정확히 같은 텐서나 중첩 배열을 받습니다. 출력이 원소 하나면 시드를 생략할 수 있습니다. `Value#backward(seed = 1.0)`는 수치 시드를 받습니다. 공유 입력에는 여러 경로의 기울기를 더하고, 반복 호출하면 리프 기울기가 누적됩니다. 새 업데이트 전에는 `optimizer.zero_grad`로 파라미터 기울기를 초기화하세요.

```ruby
require 'tensorruby'

x = TensorRuby.tensor([[[1, 2]], [[3, 4]]])
w = TensorRuby.tensor([[2], [3]])
y = x.matmul(w)
p y.shape                       # [2, 1, 1]
p y.to_a                        # [[[8.0]], [[18.0]]]

y.backward([[[1]], [[2]]])
p w.values.map(&:grad)          # [7.0, 10.0]
y.sum.backward
p w.values.map(&:grad)          # [11.0, 16.0]
```

## 오류 조건

들쭉날쭉한 배열, 빈 차원, 원소 수가 다른 명시적 형태, 불가능한 브로드캐스팅, 형태가 다른 시드는 `ArgumentError`를 일으킵니다. 범위를 벗어난 인덱스는 `IndexError`를 일으킵니다. 새 연산을 기록하지 않고 추론하려면 `TensorRuby.no_grad { … }`를 사용하세요.
