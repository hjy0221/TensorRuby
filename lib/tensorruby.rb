require_relative 'tensorruby/version'
require_relative 'tensorruby/value'
require_relative 'tensorruby/tensor'
require_relative 'tensorruby/nn'
require_relative 'tensorruby/optim'
require_relative 'tensorruby/data_loader'

module TensorRuby
  def self.tensor(data, shape: nil)
    Tensor.new(data, shape: shape)
  end
end
