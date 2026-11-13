# 開発ガイド

[ホーム](../../README.ja.md) · [クイックスタート](quickstart.md) · [API リファレンス](api.md) · [実行例](examples.md) · [開発ガイド](development.md)

[English](../en/development.md) · [日本語](../ja/development.md) · [한국어](../ko/development.md)

クローンしたリポジトリのルートで実行します。テストには Minitest が必要です。

```sh
ruby -Ilib test/tensorruby_test.rb
ruby bin/tensorruby run xor
gem build tensorruby.gemspec
SIZE=48 ITERATIONS=20 ruby -Ilib benchmark/matmul.rb
```

ベンチマークは順伝播のみと順伝播・逆伝播を合わせた時間とオブジェクト割り当て数を表示します。同じマシン、Ruby バージョン、行列サイズで比較してください。現在のランタイムは Ruby の CPU カーネルを使用します。

[開発ロードマップ](../../ROADMAP.ja.md)

## 行列積バックエンド

ブロック内の行列積に指定したバックエンドを使います。逆伝播には順伝播で選択したバックエンドを使います。独自バックエンドは `TensorRuby::CPU` と同じ平坦な配列の API を実装します。

```ruby
require "tensorruby"

TensorRuby::Backend.with(TensorRuby::Backend::PackedCPU) do
  TensorRuby.tensor([[1, 2]]).matmul([[3], [4]]).to_a # [[11.0]]
end
```

```sh
ruby -Ilib benchmark/backends.rb
```

`Backend.profile { … }` はバックエンド、順伝播・逆伝播、`[rows, inner, columns]` ごとの呼び出し回数、実測カーネル時間（秒）、推定 FLOP 数を返します。バッチ行列積は行列ごとに集計します。

macOS では `Backend::Accelerate.available?` で利用可否を確認し、`Backend.with(Backend::Accelerate)` で倍精度の行列積を選択できます。
