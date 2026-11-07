require 'minitest/autorun'
require 'tensorruby'
class TensorRubyTest < Minitest::Test
  include TensorRuby
  def test_shared_graph_and_repeated_backward
    x, y = Value.new(2), Value.new(3); z = x*x + x*y
    z.backward; assert_in_delta 7, x.grad; assert_in_delta 2, y.grad
    z.backward; assert_in_delta 14, x.grad; assert_in_delta 4, y.grad
    x.zero_grad; y.zero_grad; z.backward; assert_in_delta 7, x.grad
  end
  def test_leaf_and_forward_capture
    x = Value.new(3); x.backward(2); x.backward(3); assert_in_delta 5, x.grad
    x.zero_grad; y = x*x; x.data = 7; y.backward; assert_in_delta 6, x.grad
  end
  def test_tensor_broadcast
    x = Tensor.new([[1,2,3],[4,5,6]]); b = Tensor.new([10,20,30]); r = x+b
    assert_equal [2,3], r.shape; assert_equal [3,1], r.strides
    assert_equal [[11.0,22.0,33.0],[14.0,25.0,36.0]], r.to_a
    r.sum.backward; assert_equal [2.0,2.0,2.0], b.values.map(&:grad)
    assert_raises(ArgumentError) { x + Tensor.new([1,2]) }
    assert_raises(ArgumentError) { Tensor.new([[1],[2,3]]) }
  end
  def test_matmul
    a = Tensor.new([[1,2],[3,4]]); b = Tensor.new([[5],[6]]); r = a.matmul(b)
    assert_equal [[17.0],[39.0]], r.to_a
    r.sum.backward; assert_equal [4.0,6.0], b.values.map(&:grad)
    assert_raises(ArgumentError) { b.matmul(b) }
  end
  def test_nn_and_sgd
    m = NN::Linear.new(1,1,random: Random.new(1)); m.weight.values.first.data = 0; m.bias.values.first.data = 0
    opt = Optim::SGD.new(m.parameters, lr: 0.1)
    NN::MSELoss.new.call(m.call([[1]]), [[2]]).backward; opt.step; opt.zero_grad
    assert_in_delta 0.4, m.weight.values.first.data; assert_in_delta 0.4, m.bias.values.first.data
    assert m.parameters.all? { |p| p.grad.zero? }
    assert_equal [[0.0,2.0]], NN::ReLU.new.call([[-1,2]]).to_a
    assert_raises(ArgumentError) { NN::MSELoss.new.call([1], [[1]]) }
  end
  def test_cross_entropy_with_large_logits_and_batch_gradients
    logits = Tensor.new([[1000, 1000, 1000], [-1000, -1000, -1000]])
    loss = NN::CrossEntropyLoss.new.call(logits, [0, 2])
    assert_in_delta Math.log(3), loss.data, 1e-12
    loss.backward(2.0)
    expected = [-2.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3, -2.0/3]
    logits.values.zip(expected).each { |value, grad| assert_in_delta grad, value.grad, 1e-12 }
    assert_raises(ArgumentError) { NN::CrossEntropyLoss.new.call(logits, [0, 3]) }
  end

  def test_adam_keeps_moments_across_steps
    parameter = Value.new(1)
    optimizer = Optim::Adam.new([parameter], lr: 0.1, betas: [0.5, 0.5])
    parameter.backward
    optimizer.step
    assert_in_delta 0.9, parameter.data, 1e-8
    optimizer.zero_grad
    parameter.backward(3.0)
    optimizer.step
    expected = 0.9 - 0.1 * (7.0 / 3) / Math.sqrt(19.0 / 3)
    assert_in_delta expected, parameter.data, 1e-8
  end

  def test_tensor_pullbacks_match_finite_differences
    x = Tensor.new([[1, -2], [3, 4]])
    weight = Tensor.new([[2, -1], [1, 2]])
    bias = Tensor.new([0.3, -0.7])
    forward = -> { (x.matmul(weight) + bias).relu }
    seed = Tensor.new([[0.5, 2], [-1, 3]])
    forward.call.backward(seed)
    epsilon = 1e-5

    (x.values + weight.values + bias.values).each do |parameter|
      original = parameter.data
      parameter.data = original + epsilon
      upper = (forward.call * seed).sum.data
      parameter.data = original - epsilon
      lower = (forward.call * seed).sum.data
      parameter.data = original
      assert_in_delta((upper - lower) / (2 * epsilon), parameter.grad, 1e-8)
    end
  end

  def test_tensor_backward_accumulates_shared_and_scalar_paths
    x = Tensor.new([2, 3])
    intermediate = x * x
    loss = (intermediate + intermediate).sum + intermediate[0]
    2.times { loss.backward }
    assert_equal [24.0, 24.0], x.values.map(&:grad)
    assert_equal [3.0, 2.0], intermediate.values.map(&:grad)
    assert_raises(ArgumentError) { intermediate.backward }
    assert_raises(ArgumentError) { intermediate.backward([[1, 1]]) }
  end

  def test_data_loader_yields_paired_batches_and_keeps_remainder
    data = [[[1], 4], [[2], 5], [[3], 6]]
    loader = Data::DataLoader.new(data, batch_size: 2, shuffle: false)
    expected = [[[[1.0], [2.0]], [4.0, 5.0]], [[[3.0]], [6.0]]]

    assert_equal 2, loader.each.size
    2.times { assert_equal expected, loader.map { |x, y| [x.to_a, y.to_a] } }
    shuffled = Data::DataLoader.new(data, batch_size: 2, random: Random.new(42))
    assert_equal data, shuffled.flat_map { |x, y| x.to_a.zip(y.to_a) }.sort
    assert_empty Data::DataLoader.new([], batch_size: 2).to_a
    assert_raises(ArgumentError) { Data::DataLoader.new(data, batch_size: 0) }
  end

  def test_regression
    m = NN::Linear.new(1,1,random: Random.new(42)); opt = Optim::SGD.new(m.parameters,lr: 0.05)
    x = Tensor.new([[-2],[-1],[0],[1],[2]]); y = Tensor.new([[-3],[-1],[1],[3],[5]])
    200.times { opt.zero_grad; NN::MSELoss.new.call(m.call(x),y).backward; opt.step }
    assert_operator NN::MSELoss.new.call(m.call(x),y).data, :<, 1e-12
  end
end
