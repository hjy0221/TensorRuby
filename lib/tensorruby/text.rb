module TensorRuby
  module Text
    # Learns byte-pair merges while keeping all 256 UTF-8 byte values available.
    class BytePairTokenizer
      attr_reader :merges

      def self.train(documents, max_merges: 64)
        unless max_merges.is_a?(Integer) && max_merges >= 0
          raise ArgumentError, 'max_merges must be a nonnegative integer'
        end

        sequences = documents.map { |text| utf8_bytes(text) }
        merges = []
        max_merges.times do
          counts = Hash.new(0)
          sequences.each { |tokens| tokens.each_cons(2) { |pair| counts[pair] += 1 } }
          break if counts.empty?

          pair, count = counts.max_by { |(left, right), frequency| [frequency, -left, -right] }
          break if count < 2

          token = 256 + merges.size
          merges << pair
          sequences.map! { |tokens| merge(tokens, pair, token) }
        end
        new(merges)
      end

      def initialize(merges = [])
        seen = {}
        @merges = merges.each_with_index.map do |pair, index|
          unless pair.is_a?(Array) && pair.size == 2 &&
                 pair.all? { |id| id.is_a?(Integer) && id >= 0 && id < 256 + index } && !seen[pair]
            raise ArgumentError, 'invalid byte-pair merge'
          end

          seen[pair] = true
          pair.dup.freeze
        end.freeze
        @pieces = (0...256).map { |byte| [byte].pack('C') }
        @merges.each { |left, right| @pieces << @pieces[left] + @pieces[right] }
      end

      def size
        @pieces.size
      end

      def encode(text)
        tokens = self.class.utf8_bytes(text)
        merges.each_with_index { |pair, index| tokens = self.class.merge(tokens, pair, 256 + index) }
        tokens
      end

      def decode(ids)
        unless ids.all? { |id| id.is_a?(Integer) && id >= 0 && id < size }
          raise ArgumentError, 'token IDs must be within the tokenizer vocabulary'
        end

        ids.map { |id| @pieces[id] }.join.force_encoding(Encoding::UTF_8).scrub
      end

      def self.utf8_bytes(text)
        encoded = text.encode(Encoding::UTF_8)
        raise ArgumentError, 'text must be valid UTF-8' unless encoded.valid_encoding?

        encoded.bytes
      end

      def self.merge(tokens, pair, replacement)
        output = []
        index = 0
        while index < tokens.size
          if tokens[index] == pair[0] && tokens[index + 1] == pair[1]
            output << replacement
            index += 2
          else
            output << tokens[index]
            index += 1
          end
        end
        output
      end
    end

    class NextTokenWindows
      include Enumerable

      def initialize(documents, tokenizer:, context_size:, stride: context_size)
        unless [context_size, stride].all? { |value| value.is_a?(Integer) && value.positive? }
          raise ArgumentError, 'context size and stride must be positive integers'
        end

        @documents = documents.map { |text| tokenizer.encode(text) }
        @context_size, @stride = context_size, stride
      end

      def each
        return enum_for(:each) unless block_given?

        @documents.each do |tokens|
          (0...tokens.size - 1).step(@stride) do |offset|
            window = tokens.slice(offset, @context_size + 1)
            yield window[0...-1], window.drop(1)
          end
        end
        self
      end
    end
  end
end
