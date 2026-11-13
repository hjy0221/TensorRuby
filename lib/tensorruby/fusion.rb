module TensorRuby
  class Tensor
    # Records scalar operations to apply to each element of a single tensor.
    class ElementwiseExpression
      def initialize(instructions = [])
        @instructions = instructions.freeze
      end

      attr_reader :instructions

      def +(scalar); append(:+, scalar); end
      def -(scalar); append(:-, scalar); end
      def *(scalar); append(:*, scalar); end

      def relu
        self.class.new(instructions + [[:relu, nil]])
      end

      private

      def append(operator, scalar)
        unless scalar.is_a?(Numeric) && scalar.real? && scalar.to_f.finite?
          raise ArgumentError, 'fusion operands must be finite real scalars'
        end

        self.class.new(instructions + [[operator, scalar.to_f]])
      end
    end
    private_constant :ElementwiseExpression

    # Realizes a recorded expression in one tensor operation.
    def fuse
      expression = yield ElementwiseExpression.new
      unless expression.is_a?(ElementwiseExpression)
        raise ArgumentError, 'fusion block must return an elementwise expression'
      end

      slopes = [] if TensorRuby.grad_enabled?
      output = flat_data.map do |input|
        value = input
        slope = 1.0
        expression.instructions.each do |operator, operand|
          case operator
          when :+ then value += operand
          when :- then value -= operand
          when :*
            value *= operand
            slope *= operand if slopes
          when :relu
            unless value.positive?
              value = 0.0
              slope = 0.0
            end
          end
        end
        slopes << slope if slopes
        value
      end
      operation(shape, output, [@node]) do |gradient, pass|
        @node.accumulate_vector(pass, slopes.each_with_index.map { |slope, i| slope * gradient[i] })
      end
    end
  end
end
