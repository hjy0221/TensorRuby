# TensorRuby

[English](README.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

TensorRuby는 CPU 텐서, 자동미분, 소형 모델 학습을 위한 Pure Ruby 딥러닝 프레임워크입니다. Ruby 2.6 이상이 필요합니다.

## 설치 및 실행

```sh
curl -fL https://github.com/hjy0221/TensorRuby/releases/latest/download/tensorruby.gem -o tensorruby.gem
gem install --local tensorruby.gem
tensorruby run xor
```

## 문서

- [빠른 시작](docs/ko/quickstart.md) — gem을 설치하고 첫 모델을 학습합니다.
- [API 레퍼런스](docs/ko/api.md) — 텐서, 계층, 옵티마이저, 학습, 텍스트, 확산 API를 찾아봅니다.
- [실습 예제](docs/ko/examples.md) — 작업별로 패키지에 포함된 예제를 실행합니다.
- [개발 문서](docs/ko/development.md) — 테스트, gem 빌드, CPU 성능 측정을 실행합니다.
- [개발 로드맵](ROADMAP.ko.md)
