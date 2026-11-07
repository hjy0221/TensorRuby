module TensorRuby
  module NN
    class Module
      def parameters
        collect = lambda do |obj|
          case obj
          when Value then [obj]
          when Tensor then obj.values
          when Module then obj.instance_variables.flat_map { |v| collect.call(obj.instance_variable_get(v)) }
          when Array then obj.flat_map { |v| collect.call(v) }
          when Hash then obj.values.flat_map { |v| collect.call(v) }
          else []
          end
        end
        instance_variables.flat_map { |v| collect.call(instance_variable_get(v)) }.uniq
      end
      def zero_grad; parameters.each(&:zero_grad); self; end
    end
    class Sequential < Module
      def initialize(*layers)
        unless layers.all? { |layer| layer.is_a?(Module) && layer.respond_to?(:call) }
          raise ArgumentError, 'layers must be callable NN::Module instances'
        end

        @layers = layers.freeze
      end

      def call(input)
        @layers.reduce(input) { |output, layer| layer.call(output) }
      end
    end

    class Linear < Module
      attr_reader :weight, :bias
      def initialize(in_features, out_features, random: Random.new)
        raise ArgumentError, 'features must be positive integers' unless [in_features, out_features].all? { |n| n.is_a?(Integer) && n.positive? }
        limit = Math.sqrt(6.0 / (in_features + out_features))
        @weight = Tensor.new(Array.new(in_features) { Array.new(out_features) { random.rand * 2 * limit - limit } })
        @bias = Tensor.new(Array.new(out_features, 0.0))
      end
      def call(input); Tensor.wrap(input).matmul(weight) + bias; end
    end
    class ReLU < Module
      def call(input); Tensor.wrap(input).relu; end
    end
    # Mean cross entropy for [batch, classes] logits and class indices.
    class CrossEntropyLoss < Module
      def call(logits, target)
        logits = Tensor.wrap(logits)
        target = Tensor.wrap(target)
        batch, classes = logits.shape
        unless logits.rank == 2 && target.shape == [batch]
          raise ArgumentError, 'expected [batch, classes] logits and [batch] targets'
        end

        labels = target.to_a
        unless labels.all? { |label| label >= 0 && label < classes && label == label.to_i }
          raise ArgumentError, 'targets must be valid integer class indices'
        end

        rows = logits.to_a
        loss = 0.0
        gradients = []
        rows.zip(labels).each do |row, label|
          index = label.to_i
          scores = row
          raise ArgumentError, 'logits must be finite' unless scores.all?(&:finite?)

          maximum = scores.max
          shifted = scores.map { |score| score - maximum }
          exponentials = shifted.map { |score| Math.exp(score) }
          total = exponentials.sum
          loss += (Math.log(total) - shifted[index]) / batch
          exponentials.each_with_index do |exp, column|
            gradients << (exp / total - (column == index ? 1 : 0)) / batch
          end
        end

        logits.reduce_with_gradient(loss, gradients)
      end
    end

    class MSELoss < Module
      def call(prediction, target)
        prediction, target = Tensor.wrap(prediction), Tensor.wrap(target)
        raise ArgumentError, 'MSE shapes must match' unless prediction.shape == target.shape
        diff = prediction - target
        (diff * diff).mean
      end
    end
  end
end
