# テンソルと自動微分

[API リファレンス](api.md) · [クイックスタート](quickstart.md) · [実行例](examples.md)

[English](../en/tensor.md) · [日本語](../ja/tensor.md) · [한국어](../ko/tensor.md)

## 生成と構造の確認

- `TensorRuby.tensor(data, shape: nil)` は数値スカラーまたは整形されたネスト配列からテンソルを作ります。`shape:` を指定する場合は各次元の積が値の個数と一致する必要があります。
- `shape` は各次元の大きさ、`strides` はストレージでの要素間隔、`rank` は軸数、`size` は要素数です。
- `to_a` は数値のネスト配列を返し、0 次元の場合は数値を一つ返します。`tensor[i, j]` と `values` でスカラーの `Value` にアクセスできます。

## 演算と形状

| API | 入力と戻り値 |
| --- | --- |
| `+`, `-`, `*`, `/` | 要素ごとの演算では末尾の軸から形状を合わせ、サイズが 1 の軸を拡張します。 |
| `matmul(other)` | `[..., M, K]` と `[..., K, N]` の結果は `[..., M, N]` です。先頭の軸はブロードキャスト可能で、両方の入力が 2 次元以上である必要があります。 |
| `sum`, `mean` | 全要素の合計または平均をスカラーの `Value` として返します。 |
| `transpose` | 2 次元テンソルの二つの軸を交換したビューを返します。 |
| `permute(*axes)` | 指定した順に軸を並べたビューを返します。0 から始まる各軸番号を一度ずつ指定します。 |
| `reshape(*dimensions)` | 要素数と論理順を維持します。次元一つを `-1` で推定できます。連続入力ではストレージを共有し、非連続入力ではコピーします。 |
| `contiguous?, contiguous` | 連続入力はそのまま返します。非連続入力では逆伝播可能な行優先のコピーを返します。 |
| `softmax(causal: false)`, `normalize_rows(eps: 1e-5)` | 2 次元テンソルを受け取り、同じ形状を返します。因果 softmax は行番号より後ろの列を 0 にします。 |

ビューと元のテンソルは要素データを共有します。`view[i, j].data` を変更すると元の対応位置も変わります。ビューの勾配はビューの論理順に従います。

## 逆伝播

`Tensor#backward(seed)` は出力と同じ形状のテンソルまたはネスト配列を受け取ります。出力が 1 要素なら省略できます。`Value#backward(seed = 1.0)` は数値を受け取ります。共有入力には各経路の勾配が加算され、繰り返し呼び出すと葉の勾配が累積されます。新しい更新の前に `optimizer.zero_grad` で勾配を初期化してください。

```ruby
require 'tensorruby'

x = TensorRuby.tensor([[[1, 2]], [[3, 4]]])
w = TensorRuby.tensor([[2], [3]])
y = x.matmul(w)
p y.shape                       # [2, 1, 1]
p y.to_a                        # [[[8.0]], [[18.0]]]

y.backward([[[1]], [[2]]])
p w.values.map(&:grad)          # [7.0, 10.0]
y.sum.backward
p w.values.map(&:grad)          # [11.0, 16.0]
```

## エラー条件

不揃いな配列、空の次元、要素数が一致しない形状、ブロードキャストできない形状、出力と異なる形状のシードは `ArgumentError` になります。範囲外の添字は `IndexError` になります。`TensorRuby.no_grad { … }` を使うと新しい演算を記録せずに推論できます。
