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
