# 実行例

[ホーム](../../README.ja.md) · [クイックスタート](quickstart.md) · [API リファレンス](api.md) · [実行例](examples.md) · [開発ガイド](development.md)

[English](../en/examples.md) · [日本語](../ja/examples.md) · [한국어](../ko/examples.md)

gem のインストール後に実行してください。パスを指定しない場合、生成ファイルは現在のディレクトリに保存されます。

## 学習の基礎

一次式を学習し、XOR とミニバッチ分類を実行します。

```sh
tensorruby run regression
tensorruby run xor
tensorruby run classification
```

## 言語モデル

トークンモデルの学習、UTF-8 生テキストでの事前学習、追加学習、多言語対話と首都 QA を試せます。

```sh
tensorruby run transformer
tensorruby run gpt
tensorruby run pretrain --save pretrained.json
tensorruby run finetune --save chat-model.json pretrained.json
tensorruby run generate --interactive chat-model.json
tensorruby run chat --interactive
tensorruby run capital_chat --save capitals-weights.json
tensorruby run capital_chat --load capitals-weights.json --interactive
tensorruby run fetch_capitals capitals.json
```

## 画像生成

合成点群、白黒パターンと逆拡散過程、固定の輪郭に色の変化を加えたモンスタースプライトを生成します。

```sh
tensorruby run diffusion
tensorruby run image_diffusion samples.png trajectory.png
tensorruby run monster_diffusion all monster-sprites.png
```
