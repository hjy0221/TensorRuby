# TensorRuby

[English](README.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

TensorRuby は CPU テンソル、自動微分、小規模モデルの学習を扱う Pure Ruby の深層学習フレームワークです。Ruby 2.6 以上が必要です。

## インストールと実行

```sh
curl -fL https://github.com/hjy0221/TensorRuby/releases/latest/download/tensorruby.gem -o tensorruby.gem
gem install --local tensorruby.gem
tensorruby run xor
```

## ドキュメント

- [クイックスタート](docs/ja/quickstart.md) — gem をインストールして最初のモデルを学習します。
- [API リファレンス](docs/ja/api.md) — テンソル、レイヤー、最適化、学習、テキスト、拡散の API を調べます。
- [実行例](docs/ja/examples.md) — 目的別に同梱のサンプルを実行します。
- [開発ガイド](docs/ja/development.md) — テスト、gem のビルド、CPU 性能測定を行います。
- [開発ロードマップ](ROADMAP.ja.md)
