# 개발 문서

[홈](../../README.ko.md) · [빠른 시작](quickstart.md) · [API 레퍼런스](api.md) · [실습 예제](examples.md) · [개발 문서](development.md)

[English](../en/development.md) · [日本語](../ja/development.md) · [한국어](../ko/development.md)

복제한 저장소의 최상위 폴더에서 실행하세요. 테스트에는 Minitest가 필요합니다.

```sh
ruby -Ilib test/tensorruby_test.rb
ruby bin/tensorruby run xor
gem build tensorruby.gemspec
SIZE=48 ITERATIONS=20 ruby -Ilib benchmark/matmul.rb
```

벤치마크는 순전파와 순전파·역전파의 시간 및 객체 할당 수를 표시합니다. 같은 컴퓨터, Ruby 버전, 행렬 크기로 비교하세요. 현재 런타임은 Ruby CPU 커널을 사용합니다.

[개발 로드맵](../../ROADMAP.ko.md)

## 행렬곱 백엔드

블록 안의 행렬곱에 지정한 백엔드를 사용합니다. 역전파는 순전파에서 선택한 백엔드를 사용합니다. 자체 백엔드는 `TensorRuby::CPU`와 같은 평탄 배열 API를 구현합니다.

```ruby
require "tensorruby"

TensorRuby::Backend.with(TensorRuby::Backend::PackedCPU) do
  TensorRuby.tensor([[1, 2]]).matmul([[3], [4]]).to_a # [[11.0]]
end
```

```sh
ruby -Ilib benchmark/backends.rb
```

`Backend.profile { … }`은 백엔드, 순전파·역전파, `[rows, inner, columns]`별로 호출 횟수, 실측 커널 시간(초), 추정 FLOP 수를 반환합니다. 배치 행렬곱은 행렬별로 호출을 집계합니다.
