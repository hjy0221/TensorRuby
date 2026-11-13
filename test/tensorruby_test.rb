require 'minitest/autorun'
require 'tmpdir'
require 'tensorruby'
class TensorRubyTest < Minitest::Test
  include TensorRuby
  def test_shared_graph_and_repeated_backward
    x, y = Value.new(2), Value.new(3); z = x*x + x*y
    z.backward; assert_in_delta 7, x.grad; assert_in_delta 2, y.grad
    z.backward; assert_in_delta 14, x.grad; assert_in_delta 4, y.grad
    x.zero_grad; y.zero_grad; z.backward; assert_in_delta 7, x.grad
  end
  def test_leaf_and_forward_capture
    x = Value.new(3); x.backward(2); x.backward(3); assert_in_delta 5, x.grad
    x.zero_grad; y = x*x; x.data = 7; y.backward; assert_in_delta 6, x.grad
  end
  def test_tensor_broadcast
    x = Tensor.new([[1,2,3],[4,5,6]]); b = Tensor.new([10,20,30]); r = x+b
    assert_equal [2,3], r.shape; assert_equal [3,1], r.strides
    assert_equal [[11.0,22.0,33.0],[14.0,25.0,36.0]], r.to_a
    r.sum.backward; assert_equal [2.0,2.0,2.0], b.values.map(&:grad)
    assert_raises(ArgumentError) { x + Tensor.new([1,2]) }
    assert_raises(ArgumentError) { Tensor.new([[1],[2,3]]) }
  end
  def test_matmul
    a = Tensor.new([[1,2],[3,4]]); b = Tensor.new([[5],[6]]); r = a.matmul(b)
    assert_equal [[17.0],[39.0]], r.to_a
    r.sum.backward; assert_equal [4.0,6.0], b.values.map(&:grad)
    assert_raises(ArgumentError) { b.matmul(b) }
  end
  def test_nn_and_sgd
    m = NN::Linear.new(1,1,random: Random.new(1)); m.weight.values.first.data = 0; m.bias.values.first.data = 0
    opt = Optim::SGD.new(m.parameters, lr: 0.1)
    NN::MSELoss.new.call(m.call([[1]]), [[2]]).backward; opt.step; opt.zero_grad
    assert_in_delta 0.4, m.weight.values.first.data; assert_in_delta 0.4, m.bias.values.first.data
    assert m.parameters.all? { |p| p.grad.zero? }
    assert_equal [[0.0,2.0]], NN::ReLU.new.call([[-1,2]]).to_a
    assert_raises(ArgumentError) { NN::MSELoss.new.call([1], [[1]]) }
  end
  def test_cross_entropy_with_large_logits_and_batch_gradients
    logits = Tensor.new([[1000, 1000, 1000], [-1000, -1000, -1000]])
    loss = NN::CrossEntropyLoss.new.call(logits, [0, 2])
    assert_in_delta Math.log(3), loss.data, 1e-12
    loss.backward(2.0)
    expected = [-2.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3, -2.0/3]
    logits.values.zip(expected).each { |value, grad| assert_in_delta grad, value.grad, 1e-12 }
    assert_raises(ArgumentError) { NN::CrossEntropyLoss.new.call(logits, [0, 3]) }
  end

  def test_adam_keeps_moments_across_steps
    parameter = Value.new(1)
    optimizer = Optim::Adam.new([parameter], lr: 0.1, betas: [0.5, 0.5])
    parameter.backward
    optimizer.step
    assert_in_delta 0.9, parameter.data, 1e-8
    optimizer.zero_grad
    parameter.backward(3.0)
    optimizer.step
    expected = 0.9 - 0.1 * (7.0 / 3) / Math.sqrt(19.0 / 3)
    assert_in_delta expected, parameter.data, 1e-8
  end

  def test_tensor_pullbacks_match_finite_differences
    x = Tensor.new([[1, -2], [3, 4]])
    weight = Tensor.new([[2, -1], [1, 2]])
    bias = Tensor.new([0.3, -0.7])
    forward = -> { (x.matmul(weight) + bias).relu }
    seed = Tensor.new([[0.5, 2], [-1, 3]])
    forward.call.backward(seed)
    epsilon = 1e-5

    (x.values + weight.values + bias.values).each do |parameter|
      original = parameter.data
      parameter.data = original + epsilon
      upper = (forward.call * seed).sum.data
      parameter.data = original - epsilon
      lower = (forward.call * seed).sum.data
      parameter.data = original
      assert_in_delta((upper - lower) / (2 * epsilon), parameter.grad, 1e-8)
    end
  end

  def test_tensor_backward_accumulates_shared_and_scalar_paths
    x = Tensor.new([2, 3])
    intermediate = x * x
    loss = (intermediate + intermediate).sum + intermediate[0]
    2.times { loss.backward }
    assert_equal [24.0, 24.0], x.values.map(&:grad)
    assert_equal [3.0, 2.0], intermediate.values.map(&:grad)
    assert_raises(ArgumentError) { intermediate.backward }
    assert_raises(ArgumentError) { intermediate.backward([[1, 1]]) }
  end

  def test_data_loader_yields_paired_batches_and_keeps_remainder
    data = [[[1], 4], [[2], 5], [[3], 6]]
    loader = Data::DataLoader.new(data, batch_size: 2, shuffle: false)
    expected = [[[[1.0], [2.0]], [4.0, 5.0]], [[[3.0]], [6.0]]]

    assert_equal 2, loader.each.size
    2.times { assert_equal expected, loader.map { |x, y| [x.to_a, y.to_a] } }
    shuffled = Data::DataLoader.new(data, batch_size: 2, random: Random.new(42))
    assert_equal data, shuffled.flat_map { |x, y| x.to_a.zip(y.to_a) }.sort
    assert_empty Data::DataLoader.new([], batch_size: 2).to_a
    assert_raises(ArgumentError) { Data::DataLoader.new(data, batch_size: 0) }
  end

  def test_eval_propagates_to_dropout_and_training_uses_scaled_mask
    dropout = NN::Dropout.new(0.5, random: Random.new(42))
    model = NN::Sequential.new(NN::Sequential.new(dropout))
    x = Tensor.new([1, 1, 1, 1])

    model.eval
    refute dropout.training?
    assert_same x, model.call(x)

    model.train
    output = model.call(x)
    assert_equal [0.0, 2.0, 2.0, 2.0], output.to_a
    output.sum.backward
    assert_equal output.to_a, x.values.map(&:grad)
  end

  def test_no_grad_detaches_new_operations_and_restores_scope
    x = Tensor.new([2, 3])
    scalar = x[0]
    detached = TensorRuby.no_grad do
      TensorRuby.no_grad { refute TensorRuby.grad_enabled? }
      refute TensorRuby.grad_enabled?
      assert_empty((scalar * scalar).parents)
      x * x
    end
    detached.sum.backward
    assert_equal [0.0, 0.0], x.values.map(&:grad)
    assert_raises(RuntimeError) { TensorRuby.no_grad { raise 'failure' } }
    assert TensorRuby.grad_enabled?
    tracked = x * x
    TensorRuby.no_grad { tracked.backward([1, 1]) }
    assert_equal [4.0, 6.0], x.values.map(&:grad)
  end

  def test_gpt_is_causal_and_embedding_gradients_accumulate
    model = NN::GPT.new(vocab_size: 5, context_size: 3, features: 4, heads: 2,
                        layers: 1, random: Random.new(9))
    first = model.call([1, 2, 3])
    second = model.call([1, 2, 4])
    assert_equal [3, 5], first.shape
    assert_equal first.to_a.first(2), second.to_a.first(2)
    NN::CrossEntropyLoss.new.call(first, [2, 3, 4]).backward
    assert model.parameters.any? { |value| value.grad.abs > 1e-6 }

    embedding = Tensor.new([[1, 2], [3, 4]])
    embedding.gather_rows([1, 0, 1]).sum.backward
    assert_equal [1.0, 1.0, 2.0, 2.0], embedding.values.map(&:grad)
  end

  def test_layer_normalization_and_gelu_match_finite_differences
    x = Tensor.new([[-2, 1, 4]])
    weights = Tensor.new([[2, -1, 0.5]])
    evaluate = -> { (x.normalize_rows.gelu * weights).sum }
    evaluate.call.backward
    x.values.each do |parameter|
      value = parameter.data
      parameter.data = value + 1e-5
      upper = evaluate.call.data
      parameter.data = value - 1e-5
      lower = evaluate.call.data
      parameter.data = value
      assert_in_delta((upper - lower) / 2e-5, parameter.grad, 1e-8)
    end
  end

  def test_diffusion_schedule_noises_and_samples
    schedule = Diffusion::Schedule.new(steps: 3, beta_start: 0.1, beta_end: 0.2)
    x = schedule.add_noise([[1]], [[2]], 0)
    assert_in_delta Math.sqrt(0.9) + 2 * Math.sqrt(0.1), x.to_a[0][0], 1e-12
    assert_raises(ArgumentError) { schedule.add_noise([[1]], [2], 0) }

    visited = []
    frames = []
    observer = ->(remaining, sample) { frames << [remaining, sample.to_a] }
    output = schedule.sample([1, 2], random: Random.new(2), on_step: observer) do |_noisy, step|
      visited << step
      refute TensorRuby.grad_enabled?
      [[0, 0]]
    end
    assert_equal [2, 1, 0], visited
    assert_equal [3, 2, 1, 0], frames.map(&:first)
    assert_equal output.to_a, frames.last.last
    assert_equal [1, 2], output.shape
    assert output.to_a.flatten.all?(&:finite?)
  end

  def test_chat_supervises_only_assistant_tokens_and_turn_end
    vocab = Chat::Vocabulary.new(['안녕', '반가워요'])
    input, targets = vocab.training_example([['안녕', '반가워요']])
    assert_equal [0, 3, 1, 4], input
    assert_equal [-1, -1, 4, 2], targets

    logits = Tensor.new(Array.new(input.size) { Array.new(vocab.size, 0.0) })
    loss = NN::CrossEntropyLoss.new(ignore_index: Chat::Vocabulary::IGNORE).call(logits, targets)
    assert_in_delta Math.log(vocab.size), loss.data, 1e-12
    loss.backward
    assert logits.values.first(2 * vocab.size).all? { |value| value.grad.zero? }
    assert_in_delta(-0.4, logits[2, 4].grad, 1e-12)
  end

  def test_byte_pair_tokenizer_round_trips_unseen_utf8
    tokenizer = Text::BytePairTokenizer.train(['banana banana', 'banana'], max_merges: 8)
    assert_operator tokenizer.encode('banana').size, :<, 6
    text = "한글 日本語 English 🌱"
    assert_equal text, tokenizer.decode(tokenizer.encode(text))
    assert_equal tokenizer.encode(text), Text::BytePairTokenizer.new(tokenizer.merges).encode(text)
    assert_raises(ArgumentError) { Text::BytePairTokenizer.new([[256, 0]]) }
  end

  def test_next_token_windows_do_not_cross_document_boundaries
    tokenizer = Text::BytePairTokenizer.new
    windows = Text::NextTokenWindows.new(%w[abcde xy], tokenizer: tokenizer, context_size: 2)
    decoded = windows.map { |input, target| [tokenizer.decode(input), tokenizer.decode(target)] }
    assert_equal [%w[ab bc], %w[cd de], %w[x y]], decoded
  end

  def test_checkpoint_preserves_byte_vocabulary_and_gpt_predictions
    tokenizer = Text::BytePairTokenizer.train(['hello hello'])
    vocabulary = Chat::ByteVocabulary.new(tokenizer)
    model = NN::GPT.new(vocab_size: vocabulary.size, context_size: 8, features: 4,
                        heads: 2, layers: 1, random: Random.new(3))
    input = vocabulary.encode('hello')
    assert_equal '新語', vocabulary.decode(vocabulary.encode('新語'))
    assert input.all? { |id| id >= Chat::Vocabulary::FIRST_WORD }
    Dir.mktmpdir do |dir|
      path = File.join(dir, 'model.json')
      Text::Checkpoint.save(path, model, vocabulary)
      reloaded, reloaded_vocabulary = Text::Checkpoint.load(path)
      assert_equal input, reloaded_vocabulary.encode('hello')
      assert_equal model.call(input).to_a, reloaded.call(input).to_a
    end
  end

  def test_global_gradient_norm_clipping
    left, right = Value.new(0), Value.new(0)
    left.grad = 3
    right.grad = 4
    assert_in_delta 5, Optim.clip_grad_norm([left, right, left], 2), 1e-12
    assert_in_delta 1.2, left.grad, 1e-12
    assert_in_delta 1.6, right.grad, 1e-12
  end

  def test_trainer_weights_and_accumulates_batches_before_one_update
    model = NN::Linear.new(1, 1, random: Random.new(1))
    model.parameters.each { |parameter| parameter.data = 0.0 }
    optimizer = Optim::SGD.new(model.parameters, lr: 0.1)
    trainer = Training::Trainer.new(
      model, optimizer: optimizer, loss: NN::MSELoss.new, weight: ->(target) { target.first.first }
    )
    data = [[[[1]], [[1]]], [[[1]], [[3]]], [[[1]], [[0]]]]

    assert_in_delta 7.0, trainer.evaluate(data), 1e-12
    assert model.training?
    assert_in_delta 7.0, trainer.train_epoch(data, accumulation_steps: 2), 1e-12
    assert_equal 1, trainer.steps
    model.parameters.each { |parameter| assert_in_delta 0.5, parameter.data, 1e-12 }
    model.eval
    assert_in_delta 3.0, trainer.evaluate(data), 1e-12
    refute model.training?
  end

  def test_matmul_accumulates_both_operand_paths
    matrix = Tensor.new([[1, 2], [3, 4]])
    output = matrix.matmul(matrix)
    assert_equal [[7.0, 10.0], [15.0, 22.0]], output.to_a
    output.sum.backward
    assert_equal [7.0, 11.0, 9.0, 13.0], matrix.values.map(&:grad)
  end

  def test_batched_matmul_broadcasts_both_inputs_and_sums_gradients
    left = Tensor.new([[[[1, 2]]], [[[3, 4]]]])
    right = Tensor.new([[[[2], [3]], [[5], [7]], [[11], [13]]]])
    result = left.matmul(right)

    assert_equal [2, 3, 1, 1], result.shape
    assert_equal [[[[8.0]], [[19.0]], [[37.0]]], [[[18.0]], [[43.0]], [[85.0]]]], result.to_a
    result.backward([[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]]])
    assert_equal [45.0, 56.0, 99.0, 125.0], left.values.map(&:grad)
    assert_equal [13.0, 18.0, 17.0, 24.0, 21.0, 30.0], right.values.map(&:grad)
  end

  def test_batched_matmul_accepts_rank_two_weights
    input = Tensor.new([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    weight = Tensor.new([[2], [3]])
    output = input.matmul(weight)
    assert_equal [[[8.0], [18.0]], [[28.0], [38.0]]], output.to_a
    output.sum.backward
    assert_equal [16.0, 20.0], weight.values.map(&:grad)
    assert_raises(ArgumentError) { input.matmul(Tensor.new([[[1], [2]]] * 3)) }
  end

  def test_views_share_storage_and_keep_logical_order
    base = Tensor.new([[1, 2, 3], [4, 5, 6]])
    transposed = base.transpose
    assert_equal [1, 3], transposed.strides
    refute transposed.contiguous?
    assert_equal [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], transposed.to_a
    transposed[1, 0].data = 9
    assert_equal 9, base[0, 1].data
    reshaped = base.reshape(3, -1)
    base[0, 0].data = 8
    assert_equal 8, reshaped[0, 0].data
    copy = transposed.contiguous
    assert copy.contiguous?
    assert_same copy, copy.contiguous
    copy[1, 0].data = 10
    assert_equal 9, transposed[1, 0].data
    assert_raises(ArgumentError) { base.reshape(4, -1) }
    assert_raises(ArgumentError) { base.permute(0, 0) }
  end

  def test_transpose_reshape_matmul_backpropagates_in_original_order
    base = Tensor.new([[1, 2, 3], [4, 5, 6]])
    reordered = base.transpose.reshape([2, 3])
    result = reordered.matmul([[1], [10], [100]])
    assert_equal [[241.0], [635.0]], result.to_a
    result.backward([[2], [3]])
    assert_equal [2.0, 200.0, 30.0, 20.0, 3.0, 300.0], base.values.map(&:grad)
  end

  def test_rank_three_permutation_and_detached_view
    base = Tensor.new([[[1, 2]], [[3, 4]]])
    result = base.permute([2, 0, 1])
    assert_equal [[[1.0], [3.0]], [[2.0], [4.0]]], result.to_a
    (result + [10]).backward([[[1], [2]], [[3], [4]]])
    assert_equal [1.0, 3.0, 2.0, 4.0], base.values.map(&:grad)
    TensorRuby.no_grad { base.permute(2, 0, 1) }.sum.backward
    assert_equal [1.0, 3.0, 2.0, 4.0], base.values.map(&:grad)
  end

  def test_axis_mean_on_transposed_view_with_broadcast_backward
    source = Tensor.new([[1, 2, 3], [5, 8, 11]])
    view = source.transpose
    centered = view - view.mean(axis: -1, keepdim: true)
    assert_equal [[-2.0, 2.0], [-3.0, 3.0], [-4.0, 4.0]], centered.to_a
    centered.backward([[1, 3], [2, 6], [3, 9]])
    assert_equal [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0], source.values.map(&:grad)
  end

  def test_axis_sum_preserves_shapes_and_accumulates_gradients
    x = Tensor.new([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert_equal [14.0, 22.0], x.sum(axis: [0, -1]).to_a
    assert_equal [1, 1, 1], x.sum(keepdim: true).shape
    assert_equal [], x.sum(axis: [0, 1, 2]).shape
    x.sum(axis: [0, -1]).backward([2, 3])
    assert_equal [2.0, 2.0, 3.0, 3.0] * 2, x.values.map(&:grad)
    assert_raises(ArgumentError) { x.mean(axis: [1, -2]) }
    assert_raises(ArgumentError) { x.sum(axis: 3) }
  end

  def test_affine_relu_matches_separate_operations_on_a_view
    input = Tensor.new([[1, -1, 0], [2, 3, -2]])
    view = input.transpose
    separate = (view * -2 + 2).relu
    fused = view.affine_relu(scale: -2, bias: 2)
    assert_equal separate.to_a, fused.to_a

    seed = [[1, 2], [3, 4], [5, 6]]
    separate.backward(seed)
    expected = input.values.map(&:grad)
    input.values.each(&:zero_grad)
    fused.backward(seed)
    assert_equal expected, input.values.map(&:grad)
    assert_equal [-0.0, -6.0, -10.0, 0.0, 0.0, -12.0], expected
    assert_raises(ArgumentError) { view.affine_relu(scale: Float::INFINITY, bias: 0) }
  end

  def test_scoped_matmul_backend_captures_backward_and_matches_transformer
    original = Backend.current
    calls = []
    packed = Backend::PackedCPU
    instrumented = Object.new
    [:matmul, :matmul_backward].each do |method|
      instrumented.define_singleton_method(method) do |*args|
        calls << method
        packed.public_send(method, *args)
      end
    end
    model = NN::TransformerBlock.new(4, heads: 2, hidden_features: 8, random: Random.new(12))
    input = Tensor.new([[1, 2, 3, 4], [4, 1, 3, 2]])
    expected = model.call(input)
    expected.sum.backward
    parameters = model.parameters + input.values
    gradients = parameters.map(&:grad)
    parameters.each(&:zero_grad)

    result = Backend.with(instrumented) { model.call(input) }
    assert_same original, Backend.current
    assert_equal expected.to_a, result.to_a
    result.sum.backward
    assert_includes calls, :matmul_backward
    parameters.zip(gradients).each { |value, gradient| assert_in_delta gradient, value.grad, 1e-10 }
    Backend.with(packed) do
      assert_raises(RuntimeError) { Backend.with(original) { raise 'failure' } }
      assert_same packed, Backend.current
    end
    assert_same original, Backend.current
  end

  def test_matmul_profile_groups_batched_calls_and_restores_scope
    x = Tensor.new([[[1, 2]], [[3, 4]]])
    report = Backend.profile do
      assert_raises(RuntimeError) { Backend.profile { raise 'stop' } }
      x.matmul([[2], [3]]).sum.backward
    end
    assert_equal [[:forward, 2, 8], [:backward, 2, 16]],
                 report.map { |row| row.values_at(:phase, :calls, :estimated_flops) }
    assert report.all? { |row| row[:shape] == [1, 2, 1] && row[:seconds] >= 0 }
    assert_empty Backend.profile {}
  end

  def test_accelerate_matches_cpu_for_rectangular_batched_matmul
    backend = Backend::Accelerate
    skip 'macOS Accelerate unavailable' unless backend.available?

    left = Tensor.new([[[1, 2, 3], [4, 5, 6]], [[-2, 1, 4], [0, 3, 2]]])
    right = Tensor.new([[2, 3, 5, 7], [1, 4, 2, 6], [8, 3, 1, 9]])
    parameters = left.values + right.values
    expected = left.matmul(right)
    expected.sum.backward
    gradients = parameters.map(&:grad)
    parameters.each(&:zero_grad)

    actual = Backend.with(backend) { left.matmul(right) }
    assert_equal expected.to_a, actual.to_a
    actual.sum.backward
    assert_equal gradients, parameters.map(&:grad)
  end

  def test_regression
    m = NN::Linear.new(1,1,random: Random.new(42)); opt = Optim::SGD.new(m.parameters,lr: 0.05)
    x = Tensor.new([[-2],[-1],[0],[1],[2]]); y = Tensor.new([[-3],[-1],[1],[3],[5]])
    200.times { opt.zero_grad; NN::MSELoss.new.call(m.call(x),y).backward; opt.step }
    assert_operator NN::MSELoss.new.call(m.call(x),y).data, :<, 1e-12
  end
end
