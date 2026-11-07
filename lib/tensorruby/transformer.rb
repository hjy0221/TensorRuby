module TensorRuby
  module NN
    class LayerNorm < Module
      def initialize(features, eps: 1e-5)
        unless features.is_a?(Integer) && features.positive?
          raise ArgumentError, 'features must be a positive integer'
        end

        @scale = Tensor.new(Array.new(features, 1.0))
        @bias = Tensor.new(Array.new(features, 0.0))
        @eps = eps
      end

      def call(input)
        input = Tensor.wrap(input)
        unless input.rank == 2 && input.shape[1] == @scale.size
          raise ArgumentError, 'expected [sequence, features] input'
        end

        input.normalize_rows(eps: @eps) * @scale + @bias
      end
    end

    class GELU < Module
      def call(input)
        Tensor.wrap(input).gelu
      end
    end

    class SelfAttention < Module
      def initialize(features, heads: 1, causal: false, random: Random.new)
        unless features.is_a?(Integer) && features.positive? &&
               heads.is_a?(Integer) && heads.positive? && (features % heads).zero?
          raise ArgumentError, 'features must be positive and divisible by heads'
        end

        width = features / heads
        @projections = Array.new(heads) { Array.new(3) { Linear.new(features, width, random: random) } }
        @output = Linear.new(features, features, random: random)
        @scale = 1.0 / Math.sqrt(width)
        @causal = causal
      end

      def call(input)
        contexts = @projections.map do |query, key, value|
          scores = query.call(input).matmul(key.call(input).transpose) * @scale
          scores.softmax(causal: @causal).matmul(value.call(input))
        end
        @output.call(Tensor.concat_columns(contexts))
      end
    end

    # Pre-normalized block for one sequence of feature vectors.
    class TransformerBlock < Module
      def initialize(features, hidden_features:, heads: 1, causal: true, random: Random.new)
        @attention_norm = LayerNorm.new(features)
        @attention = SelfAttention.new(features, heads: heads, causal: causal, random: random)
        @feed_forward_norm = LayerNorm.new(features)
        @feed_forward = Sequential.new(
          Linear.new(features, hidden_features, random: random),
          GELU.new,
          Linear.new(hidden_features, features, random: random)
        )
      end

      def call(input)
        input = Tensor.wrap(input)
        residual = input + @attention.call(@attention_norm.call(input))
        residual + @feed_forward.call(@feed_forward_norm.call(residual))
      end
    end
  end
end
