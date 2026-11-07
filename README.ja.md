# TensorRuby

[English](README.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

TensorRuby は Pure Ruby で実装した小さな深層学習フレームワークです。Ruby 2.6 以上が必要で、実行時の外部依存関係はありません。

## インストール

GitHub の最新リリースから gem をインストールします。

```sh
curl -fL https://github.com/hjy0221/TensorRuby/releases/latest/download/tensorruby.gem -o tensorruby.gem
gem install --local tensorruby.gem
```

## 独自のモデルを学習

gem をインストールすると、任意のディレクトリからサンプルを確認して実行できます。自作の Ruby コードでは `require "tensorruby"` で読み込みます。Trainer には `[入力, 正解]` の組を渡します。組ごとにバッチ全体を渡せます。

```sh
tensorruby examples
tensorruby run xor
```

```ruby
require 'tensorruby'

model = TensorRuby::NN::Linear.new(1, 1)
optimizer = TensorRuby::Optim::SGD.new(model.parameters, lr: 0.05)
trainer = TensorRuby::Training::Trainer.new(
  model, optimizer: optimizer, loss: TensorRuby::NN::MSELoss.new
)
data = [[TensorRuby.tensor([[0], [1], [2]]), TensorRuby.tensor([[1], [3], [5]])]]
200.times { trainer.train_epoch(data) }
p model.call([[3]]).to_a
```

## 実行方法

```sh
ruby -Ilib test/tensorruby_test.rb
ruby bin/tensorruby run regression
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

`TensorRuby.tensor` でテンソルを作り、`NN::Sequential` で層を組み合わせます。サンプル全体は `tensorruby run xor` で実行できます。

## 多クラス分類

`NN::CrossEntropyLoss` に形状が `[batch, classes]` のロジットと、形状が `[batch]` の整数クラス番号を渡します。`Optim::Adam` は学習ステップ間でモーメントを保持します。3 クラスの例は `tensorruby run classification` で実行できます。

`Data::DataLoader.new(samples, batch_size: 8)` は `[入力, 正解]` の組をミニバッチにまとめ、エポックごとにシャッフルします。

## 学習と評価

`model.train` でドロップアウトを有効にし、`model.eval` でモデルと子レイヤーのドロップアウトを無効にします。推論を `TensorRuby.no_grad` で囲むと、逆伝播用の演算記録を省略できます。

```ruby
model.eval
predictions = TensorRuby.no_grad { model.call(x).to_a }
```

## GPT と拡散モデルのサンプル

GPT-2 形式のデコーダーは整数トークン ID を受け取り、次トークン予測を学習します。トークンと位置の埋め込み、因果マルチヘッド注意、事前レイヤー正規化、GELU、埋め込みと出力の重み共有を備えます。拡散モデルは合成点群と白黒の縞模様を学習し、画像版は生成した 6×6 の画像 8 枚を PNG に保存します。

```sh
tensorruby run gpt
tensorruby run transformer
tensorruby run diffusion
tensorruby run image_diffusion diffusion-samples.png diffusion-trajectory.png
```

## 対話とモンスターのスプライト

付属の韓国語・日本語・英語の対話で小さな対話モデルを学習できます。`--data FILE` には `[ユーザー, アシスタント]` の組を含む対話の JSON 配列を指定できます。語彙は空白区切りの学習データに限定されます。スプライトは固定の輪郭と、拡散モデルで生成した色や目の変化を組み合わせます。

```sh
tensorruby run chat --interactive
tensorruby run monster_diffusion all monster-sprites.png
```

## 多言語の首都モデルを学習して読み込む

12 か国の首都を 3 言語で学習し、保存した重みを再学習せずに読み込めます。学習済みの質問と、同じ単語を並べ替えた質問の完全一致率を表示します。

```sh
tensorruby run capital_chat --save capital-chat-weights.json
tensorruby run capital_chat --load capital-chat-weights.json --interactive
```

## 生テキストで事前学習

バイトペアトークナイザーと小さな GPT をゼロから学習します。UTF-8 の任意の文字列をトークン化できます。学習用と検証用に、1 行につき 1 文書の UTF-8 ファイルを指定します。結合規則は学習用テキストから学びます。

```sh
tensorruby run pretrain
tensorruby run pretrain --train train.txt --validation validation.txt --epochs 12
```

## チェックポイントから追加学習して生成

事前学習では複数の系列の勾配をトークン数で重み付けして累積し、勾配ノルムをクリップします。学習率はウォームアップ後にコサイン減衰させます。JSON に設定、重み、結合規則を保存し、アシスタントのトークンだけを追加学習します。

```sh
tensorruby run pretrain --save pretrained.json
tensorruby run finetune --save chat-model.json pretrained.json
tensorruby run generate --chat chat-model.json "하늘?"
```

## 今後の拡張

テンソル演算ごとにグラフのノードを一つ作り、ベクトル・ヤコビアン積を計算します。`output.backward(seed)` には出力と同じ形状の勾配を渡します。出力が 1 要素なら省略できます。CPU カーネルは現在 Ruby で動作し、将来はネイティブ実装に置き換えられます。
