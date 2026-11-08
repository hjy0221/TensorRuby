# Examples

[Home](../../README.md) · [Quickstart](quickstart.md) · [API reference](api.md) · [Examples](examples.md) · [Development](development.md)

[English](../en/examples.md) · [日本語](../ja/examples.md) · [한국어](../ko/examples.md)

Run these commands after installing the gem. Commands that produce files write to the current directory unless a path is supplied.

## Foundations

Learn a linear equation, classify XOR, and train a mini-batch classifier.

```sh
tensorruby run regression
tensorruby run xor
tensorruby run classification
```

## Language models

Train a token model, pretrain on UTF-8 text, fine-tune a checkpoint, or try the small multilingual dialogue and capital demos.

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

## Image generation

Generate synthetic points, grayscale patterns and a reverse-process contact sheet, or color variations of fixed monster silhouettes.

```sh
tensorruby run diffusion
tensorruby run image_diffusion samples.png trajectory.png
tensorruby run monster_diffusion all monster-sprites.png
```
