# クイックスタート

[ホーム](../../README.ja.md) · [クイックスタート](quickstart.md) · [API リファレンス](api.md) · [実行例](examples.md) · [開発ガイド](development.md)

[English](../en/quickstart.md) · [日本語](../ja/quickstart.md) · [한국어](../ko/quickstart.md)

ターミナルで以下を実行してください。サンプルは gem に含まれており、任意のディレクトリから実行できます。

```sh
curl -fL https://github.com/hjy0221/TensorRuby/releases/latest/download/tensorruby.gem -o tensorruby.gem
gem install --local tensorruby.gem
tensorruby run xor
```

## 最初のモデルを学習

以下を `train.rb` として保存し、`ruby train.rb` で実行します。モデルは `y = 2x + 1` を学習します。Trainer に渡す各組は入力バッチと正解バッチです。

```ruby
require 'tensorruby'

x = TensorRuby.tensor([[0], [1], [2]])
y = TensorRuby.tensor([[1], [3], [5]])
model = TensorRuby::NN::Linear.new(1, 1, random: Random.new(42))
optimizer = TensorRuby::Optim::SGD.new(model.parameters, lr: 0.05)
trainer = TensorRuby::Training::Trainer.new(
  model, optimizer: optimizer, loss: TensorRuby::NN::MSELoss.new
)

data = [[x, y]]
200.times { trainer.train_epoch(data) }
p trainer.evaluate(data)
model.eval
p TensorRuby.no_grad { model.call([[3]]).to_a } # approximately [[7.0]]
```

## 次のステップ

`tensorruby examples` で全サンプルを確認できます。コマンドが見つからない場合は `gem environment` で gem の実行ファイルのディレクトリを確認し、PATH に追加します。
