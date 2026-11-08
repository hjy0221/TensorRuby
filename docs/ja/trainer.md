# Trainer とミニバッチ

[API リファレンス](api.md) · [クイックスタート](quickstart.md) · [実行例](examples.md)

[English](../en/trainer.md) · [日本語](../ja/trainer.md) · [한국어](../ko/trainer.md)

## 設定

`TensorRuby::Training::Trainer.new(model, optimizer:, loss:, weight: nil, max_grad_norm: nil, forward: nil)`

- `optimizer:` は `zero_grad` と `step` を実装します。`loss:` は予測値と正解を受け取り、有限なスカラーの `Value` を返します。
- `weight:` の既定値は入力・正解の組ごとに 1 です。要素平均では正解の要素数、マスク付きの言語モデル損失では学習対象トークン数で重み付けできます。重み 0 の組はスキップします。
- `max_grad_norm:` は更新前に勾配全体のノルムを制限します。`forward:` には `model.call(input)` の代わりに実行する関数を指定できます。

## 学習と評価

| API | 入力と戻り値 |
| --- | --- |
| `train_epoch(examples, accumulation_steps: 1, learning_rate: nil)` | 学習モードに切り替え、重み付き平均損失を返します。最大 `accumulation_steps` 組ごとに一度更新し、最後の端数も処理します。 |
| `evaluate(examples)` | 勾配を記録せず重み付き平均損失を返し、評価後は元のモードに戻します。 |
| `steps` | 過去のエポックを含め、完了したオプティマイザー更新の回数を返します。 |

`learning_rate:` は 0 始まりの更新番号から学習率を返す関数です。オプティマイザーが `step(lr:)` に対応している必要があります。`Optim::Adam` は対応しています。

## 実行例

ローダーは 2 件と 1 件のバッチを返します。正解の要素数で重み付けすると各観測値の影響が等しくなり、両バッチを累積すると更新は一度になります。

```ruby
require 'tensorruby'

samples = [[[0], [1]], [[1], [3]], [[2], [5]]]
loader = TensorRuby::Data::DataLoader.new(samples, batch_size: 2, shuffle: false)
model = TensorRuby::NN::Linear.new(1, 1, random: Random.new(42))
optimizer = TensorRuby::Optim::SGD.new(model.parameters, lr: 0.05)
trainer = TensorRuby::Training::Trainer.new(
  model, optimizer: optimizer, loss: TensorRuby::NN::MSELoss.new,
  weight: ->(target) { target.size }
)

trainer.train_epoch(loader, accumulation_steps: 2)
p trainer.steps                 # 1
300.times { trainer.train_epoch(loader, accumulation_steps: 2) }
p trainer.evaluate(loader)      # approximately 0.0
```

## エラー条件

累積回数が正でない場合、重みが負または有限でない場合、損失が有限でない場合、正の重みを持つデータがない場合は `ArgumentError` になります。`no_grad` 内で学習した場合も同じエラーになります。
