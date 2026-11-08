# API リファレンス

[ホーム](../../README.ja.md) · [クイックスタート](quickstart.md) · [API リファレンス](api.md) · [実行例](examples.md) · [開発ガイド](development.md)

[English](../en/api.md) · [日本語](../ja/api.md) · [한국어](../ko/api.md)

特に断りがない限り、以下の API は `TensorRuby` 名前空間に属します。

## テンソルと自動微分

`TensorRuby.tensor(data, shape: nil)`

- 数値やネストした配列からテンソルを作ります。平坦な配列には `shape:` を指定できます。
- `shape`、`strides`、`rank`、`size` で構造を確認できます。`to_a` は数値を取得し、`tensor[i, j]` と `values` はスカラーの `Value` を返します。
- 要素ごとの `+`、`-`、`*`、`/` は末尾の軸を基準にブロードキャストします。`matmul(other)` は 2 次元以上の入力の末尾 2 軸で行列積を計算し、先頭の軸をブロードキャストします。`[2, 3, 4] × [4, 5]` の結果は `[2, 3, 5]` です。
- `sum` と `mean` は全要素を集約して `Value` を返します。`transpose`、`softmax(causal: false)`、`normalize_rows(eps: 1e-5)`、`gather_rows(indices)`、`concat_columns(other)` は行列に適用します。
- `Value#backward(seed = 1.0)` は葉の勾配を累積します。`Tensor#backward(seed)` は出力と同じ形状の勾配を受け取り、1 要素の場合は省略できます。`TensorRuby.no_grad { … }` は新しい演算の記録を無効にします。

```ruby
require 'tensorruby'

x = TensorRuby.tensor([[1, 2], [3, 4]])
w = TensorRuby.tensor([[2], [1]])
loss = x.matmul(w).sum
loss.backward
p w.values.map(&:grad) # [4.0, 6.0]
```

## ニューラルネットワーク

- `NN::Module#parameters` は学習可能なスカラーを集めます。`train` と `eval` は子レイヤーにも適用されます。
- `NN::Sequential.new(*layers)` はレイヤーを順に実行します。`NN::Linear.new(in_features, out_features)` は末尾の軸に特徴量を持つ 2 次元以上の入力を受け取ります。
- `NN::ReLU`、`NN::GELU`、`NN::LayerNorm.new(features)`、`NN::Dropout.new(p = 0.5)` を利用できます。LayerNorm は行列を受け取り、Dropout は学習モードで動作します。
- `NN::MSELoss#call(prediction, target)` には同じ形状を渡します。`NN::CrossEntropyLoss.new(ignore_index: nil)` は `[batch, classes]` のロジットと `[batch]` のクラス番号を受け取ります。

## 学習とデータ

- `Data::DataLoader.new(dataset, batch_size:, shuffle: true, random: Random.new)` は添字で参照できるデータセットからテンソルの組を返します。最後の端数バッチも含みます。
- `Optim::SGD.new(parameters, lr:)` と `Optim::Adam.new(parameters, lr: 0.001)` は `zero_grad` と `step` を提供します。Adam では `step(lr:)` も使えます。
- `Training::Trainer.new(model, optimizer:, loss:, weight: nil, max_grad_norm: nil, forward: nil)` に `[入力, 正解]` の組を渡します。`train_epoch(examples, accumulation_steps: 1, learning_rate: nil)` は平均損失を返し、`evaluate(examples)` は勾配を記録せず評価します。学習率関数は 0 から始まる更新ステップを受け取ります。

## テキストと生成

- `NN::SelfAttention` と `NN::TransformerBlock` は 1 個の `[sequence, features]` 行列を受け取ります。`NN::GPT.new(vocab_size:, context_size:, features:, heads:, layers:)` は整数トークン系列から `[sequence, vocabulary]` のロジットを返します。
- `GPT#generate(ids, max_new_tokens:, stop_token: nil)` は各ステップで最高スコアのトークンを選び、入力と生成結果の ID をまとめて返します。
- `Text::BytePairTokenizer.train(documents, max_merges: 64)` は UTF-8 のバイト結合を学習します。`Text::NextTokenWindows` は入力と 1 トークンずらした正解を返します。
- `Text::Checkpoint.save(path, model, vocabulary)` は設定、バイト結合規則、重みを保存します。`load(path)` はモデルとバイト語彙を返します。`Chat::Session#reply` は対話をトークン化し、テキストで応答を返します。

## 拡散モデル

- `Diffusion::Schedule.new(steps:, beta_start: 0.0001, beta_end: 0.02)` は線形の DDPM スケジュールを作成します。
- `add_noise(clean, noise, step)` のステップは 0 から始まります。`sample(shape, on_step: nil) { |noisy, step| … }` のブロックは同じ形状のノイズ予測を返します。コールバックには残りステップ数と現在のサンプルが渡されます。
