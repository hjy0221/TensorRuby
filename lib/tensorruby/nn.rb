module TensorRuby
  module NN
    class Module
      def training?
        @training != false
      end

      def train(mode = true)
        unless mode == true || mode == false
          raise ArgumentError, 'training mode must be true or false'
        end

        set_training_mode(mode, {}.compare_by_identity)
        self
      end

      def eval
        train(false)
      end

      protected

      def set_training_mode(mode, visited)
        return if visited[self]

        visited[self] = true
        @training = mode
        instance_variables.each { |name| update_child_mode(instance_variable_get(name), mode, visited) }
      end

      private

      def update_child_mode(child, mode, visited)
        case child
        when Module
          child.set_training_mode(mode, visited)
        when Array, Hash
          return if visited[child]

          visited[child] = true
          items = child.is_a?(Hash) ? child.values : child
          items.each { |item| update_child_mode(item, mode, visited) }
        end
      end

      public

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
    class Dropout < Module
      def initialize(p = 0.5, random: Random.new)
        unless p.is_a?(Numeric) && p.real? && p >= 0 && p < 1
          raise ArgumentError, 'dropout probability must be in [0, 1)'
        end

        @p = p.to_f
        @random = random
      end

      def call(input)
        input = Tensor.wrap(input)
        return input unless training? && @p.positive?

        scale = 1.0 / (1.0 - @p)
        mask = Array.new(input.size) { @random.rand < @p ? 0.0 : scale }
        input * Tensor.new(mask, shape: input.shape)
      end
    end

    # Mean cross entropy for [batch, classes] logits and class indices.
    class CrossEntropyLoss < Module
      def initialize(ignore_index: nil)
        @ignore_index = ignore_index
      end

      def call(logits, target)
        logits = Tensor.wrap(logits)
        target = Tensor.wrap(target)
        batch, classes = logits.shape
        unless logits.rank == 2 && target.shape == [batch]
          raise ArgumentError, 'expected [batch, classes] logits and [batch] targets'
        end

        labels = target.to_a
        unless labels.all? { |label| label == @ignore_index || (label >= 0 && label < classes && label == label.to_i) }
          raise ArgumentError, 'targets must be valid integer class indices'
        end

        active = labels.count { |label| label != @ignore_index }
        raise ArgumentError, 'at least one target must be supervised' if active.zero?

        rows = logits.to_a
        loss = 0.0
        gradients = []
        rows.zip(labels).each do |row, label|
          if label == @ignore_index
            gradients.concat(Array.new(classes, 0.0))
            next
          end

          index = label.to_i
          scores = row
          raise ArgumentError, 'logits must be finite' unless scores.all?(&:finite?)

          maximum = scores.max
          shifted = scores.map { |score| score - maximum }
          exponentials = shifted.map { |score| Math.exp(score) }
          total = exponentials.sum
          loss += (Math.log(total) - shifted[index]) / active
          exponentials.each_with_index do |exp, column|
            gradients << (exp / total - (column == index ? 1 : 0)) / active
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
