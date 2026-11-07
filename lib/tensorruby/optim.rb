module TensorRuby
  module Optim
    class SGD
      def initialize(parameters, lr:)
        raise ArgumentError, 'learning rate must be positive and finite' unless lr.is_a?(Numeric) && lr.finite? && lr.positive?
        @parameters = parameters.to_a.uniq
        raise TypeError, 'parameters must be Values' unless @parameters.all? { |p| p.is_a?(Value) }
        @lr = lr.to_f
      end
      def zero_grad; @parameters.each(&:zero_grad); self; end
      def step; @parameters.each { |p| p.data -= @lr * p.grad }; self; end
    end
  end
end
