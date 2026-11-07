require_relative 'tensorruby/version'
require_relative 'tensorruby/value'
require_relative 'tensorruby/tensor'
require_relative 'tensorruby/nn'
require_relative 'tensorruby/transformer'
require_relative 'tensorruby/gpt'
require_relative 'tensorruby/diffusion'
require_relative 'tensorruby/optim'
require_relative 'tensorruby/data_loader'

module TensorRuby
  def self.no_grad
    previous = Thread.current[:tensorruby_no_grad]
    Thread.current[:tensorruby_no_grad] = true
    yield
  ensure
    Thread.current[:tensorruby_no_grad] = previous
  end

  def self.grad_enabled?
    !Thread.current[:tensorruby_no_grad]
  end

  def self.tensor(data, shape: nil)
    Tensor.new(data, shape: shape)
  end
end
