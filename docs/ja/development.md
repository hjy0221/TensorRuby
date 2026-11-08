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
