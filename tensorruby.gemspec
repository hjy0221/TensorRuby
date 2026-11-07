require_relative 'lib/tensorruby/version'

Gem::Specification.new do |spec|
  spec.name = 'tensorruby'
  spec.version = TensorRuby::VERSION
  spec.homepage = 'https://github.com/hjy0221/TensorRuby'
  spec.summary = 'Pure Ruby automatic differentiation and neural networks'
  spec.authors = ['TensorRuby contributors']
  spec.license = 'MIT'
  spec.files = Dir['lib/**/*.rb', 'README*.md', 'examples/**/*.rb', 'examples/data/*.json'] + ['bin/tensorruby']
  spec.bindir = 'bin'
  spec.executables = ['tensorruby']
  spec.require_paths = ['lib']
  spec.required_ruby_version = '>= 2.6'
end
