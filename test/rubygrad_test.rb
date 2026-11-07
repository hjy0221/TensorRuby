require 'minitest/autorun'
require 'rubygrad'
class RubygradTest < Minitest::Test
  include Rubygrad
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
  def test_regression
    m = NN::Linear.new(1,1,random: Random.new(42)); opt = Optim::SGD.new(m.parameters,lr: 0.05)
    x = Tensor.new([[-2],[-1],[0],[1],[2]]); y = Tensor.new([[-3],[-1],[1],[3],[5]])
    200.times { opt.zero_grad; NN::MSELoss.new.call(m.call(x),y).backward; opt.step }
    assert_operator NN::MSELoss.new.call(m.call(x),y).data, :<, 1e-12
  end
end
