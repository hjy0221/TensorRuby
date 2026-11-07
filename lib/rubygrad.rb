require_relative 'rubygrad/version'
require_relative 'rubygrad/value'
require_relative 'rubygrad/tensor'
require_relative 'rubygrad/nn'
require_relative 'rubygrad/optim'

module Rubygrad
  def self.tensor(data, shape: nil)
    Tensor.new(data, shape: shape)
  end
end
