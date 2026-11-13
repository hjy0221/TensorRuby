# Development

[Home](../../README.md) · [Quickstart](quickstart.md) · [API reference](api.md) · [Examples](examples.md) · [Development](development.md)

[English](../en/development.md) · [日本語](../ja/development.md) · [한국어](../ko/development.md)

Run these commands from a clone of the repository. Tests require Minitest.

```sh
ruby -Ilib test/tensorruby_test.rb
ruby bin/tensorruby run xor
gem build tensorruby.gemspec
SIZE=48 ITERATIONS=20 ruby -Ilib benchmark/matmul.rb
```

The benchmark reports inference and forward-plus-backward time and allocation counts. Compare revisions on the same machine, Ruby version, and matrix size. The runtime currently uses Ruby CPU kernels.

[Roadmap](../../ROADMAP.md)

## Matrix multiplication backends

The block switches the backend for matrix multiplication only. Backpropagation uses the backend selected during the forward pass. Custom backends implement the same flat-array methods as `TensorRuby::CPU`.

```ruby
require "tensorruby"

TensorRuby::Backend.with(TensorRuby::Backend::PackedCPU) do
  TensorRuby.tensor([[1, 2]]).matmul([[3], [4]]).to_a # [[11.0]]
end
```

```sh
ruby -Ilib benchmark/backends.rb
```

`Backend.profile { … }` returns rows grouped by backend, phase, and `[rows, inner, columns]`. Each row includes call count, measured kernel seconds, and estimated FLOPs. Batched calls count each matrix product separately.

On macOS, check `Backend::Accelerate.available?` and select double-precision matrix multiplication with `Backend.with(Backend::Accelerate)`.
