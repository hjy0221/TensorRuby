module TensorRuby
  module NN
    # GPT-2-style decoder for one sequence of pre-tokenized integer IDs.
    class GPT < Module
      def initialize(vocab_size:, context_size:, features:, heads:, layers:, random: Random.new)
        unless [vocab_size, context_size, features, layers].all? { |n| n.is_a?(Integer) && n.positive? }
          raise ArgumentError, 'vocabulary, context, features, and layers must be positive integers'
        end

        @context_size = context_size
        @tokens = embedding(vocab_size, features, random)
        @positions = embedding(context_size, features, random)
        @blocks = Sequential.new(*Array.new(layers) do
          TransformerBlock.new(features, heads: heads, hidden_features: 4 * features, random: random)
        end)
        @norm = LayerNorm.new(features)
      end

      def call(ids)
        unless ids.is_a?(Array) && !ids.empty? && ids.size <= @context_size
          raise ArgumentError, 'expected one nonempty token sequence within context size'
        end

        input = @tokens.gather_rows(ids) + @positions.gather_rows((0...ids.size).to_a)
        @norm.call(@blocks.call(input)).matmul(@tokens.transpose)
      end

      # Greedy generation. Context is cropped to the most recent tokens.
      def generate(ids, max_new_tokens:)
        unless max_new_tokens.is_a?(Integer) && max_new_tokens >= 0
          raise ArgumentError, 'max_new_tokens must be a nonnegative integer'
        end
        unless ids.is_a?(Array) && !ids.empty? &&
               ids.all? { |id| id.is_a?(Integer) && id >= 0 && id < @tokens.shape[0] }
          raise ArgumentError, 'expected a nonempty sequence of valid token IDs'
        end

        result = ids.dup
        TensorRuby.no_grad do
          max_new_tokens.times do
            scores = call(result.last(@context_size)).to_a.last
            result << scores.each_index.max_by { |index| scores[index] }
          end
        end
        result
      end

      private

      def embedding(rows, columns, random)
        Tensor.new(Array.new(rows) { Array.new(columns) { (random.rand - 0.5) * 0.2 } })
      end
    end
  end
end
