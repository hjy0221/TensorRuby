# TensorRuby

[English](README.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

TensorRuby は Pure Ruby で実装した小さな深層学習フレームワークです。Ruby 2.6 以上が必要で、実行時の外部依存関係はありません。

## インストール

GitHub の最新リリースから gem をインストールします。

```sh
curl -fL https://github.com/hjy0221/TensorRuby/releases/latest/download/tensorruby.gem -o tensorruby.gem
gem install --local tensorruby.gem
```

## 実行方法

```sh
ruby -Ilib test/tensorruby_test.rb
ruby examples/regression.rb
gem build tensorruby.gemspec
```

プロジェクトのルートで実行してください。テストには Minitest が必要です。Rake がインストールされていれば、`rake test` でも実行できます。

## 使い方

インストール後、以下のコードを `ruby` で実行できます。

```ruby
require 'tensorruby'
x = TensorRuby::Tensor.new([[1, 2], [3, 4]])
w = TensorRuby::Tensor.new([[2], [1]])
output = x.matmul(w) + 1
output.sum.backward
p output.to_a            # [[5.0], [11.0]]
p w.values.map(&:grad)  # [4.0, 6.0]
```

`Value#backward(seed = 1.0)` は計算グラフをトポロジカル順序の逆順にたどり、共有ノードへの勾配の寄与を加算します。呼び出すたびに中間ノードの勾配を初期化し、葉ノードの勾配は累積します。各学習ステップの前に `optimizer.zero_grad` を呼び出し、パラメータ更新後は順伝播をやり直してください。

`Tensor` は行優先順の連続領域を使い、形状とストライドを保持します。添字や `values` でスカラーの `Value` を取得できます。要素ごとの `+`、`-`、`*`、`/` は末尾の軸を基準にブロードキャストします。`matmul` は 2 次元行列に対応します。`sum` と `mean` は全要素を集約して `Value` を返します。`Linear` の入力形状は `[batch, features]` です。MSE では予測値と正解値の形状が一致している必要があります。現行 API は各軸の長さが正のテンソルと全要素の集約に対応します。

## XOR モデルの学習

`TensorRuby.tensor` でテンソルを作り、`NN::Sequential` で層を組み合わせます。サンプル全体は `ruby examples/xor.rb` で実行できます。

```ruby
require 'tensorruby'

nn = TensorRuby::NN
random = Random.new(42)
x = TensorRuby.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
target = TensorRuby.tensor([[0], [1], [1], [0]])

model = nn::Sequential.new(
  nn::Linear.new(2, 8, random: random),
  nn::ReLU.new,
  nn::Linear.new(8, 1, random: random)
)
criterion = nn::MSELoss.new
optimizer = TensorRuby::Optim::SGD.new(model.parameters, lr: 0.1)

500.times do
  optimizer.zero_grad
  loss = criterion.call(model.call(x), target)
  loss.backward
  optimizer.step
end
```

## 多クラス分類

`NN::CrossEntropyLoss` に形状が `[batch, classes]` のロジットと、形状が `[batch]` の整数クラス番号を渡します。`Optim::Adam` は学習ステップ間でモーメントを保持します。3 クラスの例は `ruby examples/classification.rb` で実行できます。

`Data::DataLoader.new(samples, batch_size: 8)` は `[入力, 正解]` の組をミニバッチにまとめ、エポックごとにシャッフルします。

## 学習と評価

`model.train` でドロップアウトを有効にし、`model.eval` でモデルと子レイヤーのドロップアウトを無効にします。推論を `TensorRuby.no_grad` で囲むと、逆伝播用の演算記録を省略できます。

```ruby
model.eval
predictions = TensorRuby.no_grad { model.call(x).to_a }
```

## 今後の拡張

テンソル演算ごとにグラフのノードを一つ作り、ベクトル・ヤコビアン積を計算します。`output.backward(seed)` には出力と同じ形状の勾配を渡します。出力が 1 要素なら省略できます。CPU カーネルは現在 Ruby で動作し、将来はネイティブ実装に置き換えられます。
