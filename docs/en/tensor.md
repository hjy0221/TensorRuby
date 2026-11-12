# Tensor and autograd

[API reference](api.md) · [Quickstart](quickstart.md) · [Examples](examples.md)

[English](../en/tensor.md) · [日本語](../ja/tensor.md) · [한국어](../ko/tensor.md)

## Construction and inspection

- `TensorRuby.tensor(data, shape: nil)` creates a tensor from a numeric scalar or rectangular nested array. With an explicit shape, the number of values must match the product of its dimensions.
- `shape` lists dimensions, `strides` gives storage steps in elements, `rank` counts axes, and `size` counts elements.
- `to_a` returns nested numeric arrays, or a number for rank zero. `tensor[i, j]` and `values` expose scalar `Value` objects in row-major order.

## Operations and shapes

| API | Input and return value |
| --- | --- |
| `+`, `-`, `*`, `/` | Elementwise operations align shapes from the right and expand axes of size one. |
| `matmul(other)` | `[..., M, K]` and `[..., K, N]` produce `[..., M, N]`. The leading shapes must broadcast. Both inputs need rank 2 or higher. |
| `sum`, `mean` | Both return a scalar `Value` computed across all elements. |
| `transpose` | Returns a rank-2 view with its two axes swapped. |
| `permute(*axes)` | Returns a view with axes in the specified order. Provide each zero-based axis exactly once. |
| `reshape(*dimensions)` | Preserves element count and logical order. One dimension may be `-1` for inference. Shares storage for contiguous inputs and copies noncontiguous inputs. |
| `contiguous?, contiguous` | Returns the original tensor if contiguous. Otherwise returns a row-major copy that supports backpropagation. |
| `softmax(causal: false)`, `normalize_rows(eps: 1e-5)` | Both take rank 2 and return the same shape. Causal softmax sets entries above the row index to zero. |

Views share element data with their source. Writing through `view[i, j].data` changes the source at the corresponding position. View gradients follow the view’s logical order.

## Backpropagation

`Tensor#backward(seed)` accepts a tensor or nested array with the exact output shape. A one-element output can omit the seed. `Value#backward(seed = 1.0)` accepts a numeric seed. Shared inputs receive the sum of their contributions, and repeated calls accumulate leaf gradients. Clear parameter gradients with `optimizer.zero_grad` before a new update.

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

## Errors

Ragged arrays, empty dimensions, mismatched explicit shapes, incompatible broadcasts, and invalid seed shapes raise `ArgumentError`. Out-of-bounds element indices raise `IndexError`. Use `TensorRuby.no_grad { … }` when you want inference without recording new operations.
