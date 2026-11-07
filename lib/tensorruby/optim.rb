module TensorRuby
  module Optim
    def self.clip_grad_norm(parameters, max_norm)
      unless max_norm.is_a?(Numeric) && max_norm.real? && max_norm.finite? && max_norm.positive?
        raise ArgumentError, 'max_norm must be positive and finite'
      end

      parameters = parameters.to_a.uniq
      norm = parameters.reduce(0.0) { |total, parameter| Math.hypot(total, parameter.grad) }
      raise ArgumentError, 'gradient norm is nonfinite' unless norm.finite?

      if norm > max_norm
        scale = max_norm / norm
        parameters.each { |parameter| parameter.grad *= scale }
      end
      norm
    end

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

    class Adam
      def initialize(parameters, lr: 0.001, betas: [0.9, 0.999], eps: 1e-8)
        unless [lr, eps].all? { |number| number.is_a?(Numeric) && number.real? && number.finite? && number.positive? }
          raise ArgumentError, 'lr and eps must be positive and finite'
        end
        unless betas.is_a?(Array) && betas.size == 2 &&
               betas.all? { |beta| beta.is_a?(Numeric) && beta.real? && beta >= 0 && beta < 1 }
          raise ArgumentError, 'betas must be two numbers in [0, 1)'
        end

        @parameters = parameters.to_a.uniq
        raise TypeError, 'parameters must be Values' unless @parameters.all? { |p| p.is_a?(Value) }

        @lr, @eps = lr.to_f, eps.to_f
        @beta1, @beta2 = betas.map(&:to_f)
        @first = Array.new(@parameters.size, 0.0)
        @second = Array.new(@parameters.size, 0.0)
        @step = 0
      end

      def zero_grad
        @parameters.each(&:zero_grad)
        self
      end

      def step(lr: @lr)
        unless lr.is_a?(Numeric) && lr.real? && lr.finite? && lr.positive?
          raise ArgumentError, 'learning rate must be positive and finite'
        end

        @step += 1
        first_correction = 1 - @beta1**@step
        second_correction = 1 - @beta2**@step

        @parameters.each_with_index do |parameter, i|
          grad = parameter.grad
          @first[i] = @beta1 * @first[i] + (1 - @beta1) * grad
          @second[i] = @beta2 * @second[i] + (1 - @beta2) * grad * grad
          mean = @first[i] / first_correction
          variance = @second[i] / second_correction
          parameter.data -= lr * mean / (Math.sqrt(variance) + @eps)
        end
        self
      end
    end
  end
end
