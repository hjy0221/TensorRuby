# 실습 예제

[홈](../../README.ko.md) · [빠른 시작](quickstart.md) · [API 레퍼런스](api.md) · [실습 예제](examples.md) · [개발 문서](development.md)

[English](../en/examples.md) · [日本語](../ja/examples.md) · [한국어](../ko/examples.md)

gem을 설치한 뒤 실행하세요. 경로를 따로 지정하지 않으면 출력 파일은 현재 폴더에 저장됩니다.

## 기초 학습

선형 식을 학습하고, XOR와 미니배치 분류를 실행합니다.

```sh
tensorruby run regression
tensorruby run xor
tensorruby run classification
```

## 언어 모델

토큰 모델 학습, UTF-8 원문 사전학습, 체크포인트 미세조정, 다국어 대화와 수도 QA를 실행합니다.

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

## 이미지 생성

합성 점, 흑백 패턴과 역확산 과정, 고정된 실루엣에 색 변형을 적용한 몬스터 스프라이트를 생성합니다.

```sh
tensorruby run diffusion
tensorruby run image_diffusion samples.png trajectory.png
tensorruby run monster_diffusion all monster-sprites.png
```
