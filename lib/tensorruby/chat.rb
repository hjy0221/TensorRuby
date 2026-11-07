module TensorRuby
  module Chat
    # A whitespace vocabulary for small, controlled chat datasets.
    class Vocabulary
      USER = 0
      ASSISTANT = 1
      END_TURN = 2
      IGNORE = -1
      FIRST_WORD = 3

      def initialize(texts)
        @words = texts.flat_map(&:split).uniq.freeze
        @ids = @words.each_with_index.to_h { |word, index| [word, index + FIRST_WORD] }.freeze
      end

      def size
        @words.size + FIRST_WORD
      end

      def encode(text)
        words = text.split
        raise ArgumentError, 'message must contain at least one word' if words.empty?

        words.map { |word| @ids.fetch(word) { raise ArgumentError, "unknown word: #{word}" } }
      end

      def decode(ids)
        ids.take_while { |id| id >= FIRST_WORD }.map { |id| @words.fetch(id - FIRST_WORD) }.join(' ')
      end

      # Each turn is a [user_message, assistant_message] pair.
      def training_example(turns)
        tokens, targets = [], []
        turns.each do |user, assistant|
          prompt = [USER, *encode(user), ASSISTANT]
          reply = [*encode(assistant), END_TURN]
          tokens.concat(prompt).concat(reply)
          targets.concat(Array.new(prompt.size, IGNORE)).concat(reply)
        end
        raise ArgumentError, 'at least one dialogue turn is required' if tokens.empty?

        [tokens[0...-1], targets.drop(1)]
      end
    end

    class Session
      def initialize(model, vocabulary, max_reply_tokens: 12)
        unless max_reply_tokens.is_a?(Integer) && max_reply_tokens.positive?
          raise ArgumentError, 'max_reply_tokens must be a positive integer'
        end

        @model, @vocabulary, @max_reply_tokens = model, vocabulary, max_reply_tokens
        reset
      end

      def reset
        @history = []
        self
      end

      def reply(message)
        prompt = [*@history, Vocabulary::USER, *@vocabulary.encode(message), Vocabulary::ASSISTANT]
        tokens = @model.generate(prompt, max_new_tokens: @max_reply_tokens, stop_token: Vocabulary::END_TURN)
        response = tokens.drop(prompt.size).take_while { |id| id >= Vocabulary::FIRST_WORD }
        @history = [*prompt, *response, Vocabulary::END_TURN]
        @vocabulary.decode(response)
      end
    end
  end
end
