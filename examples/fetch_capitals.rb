require 'json'
require 'net/http'
require 'uri'

def entities(ids, properties)
  uri = URI('https://www.wikidata.org/w/api.php')
  uri.query = URI.encode_www_form(action: 'wbgetentities', ids: ids.join('|'),
                                  props: properties, languages: 'ko|ja|en', format: 'json')
  request = Net::HTTP::Get.new(uri)
  request['User-Agent'] = 'TensorRuby educational dataset builder/1.0 (https://github.com/hjy0221/TensorRuby)'
  response = Net::HTTP.start(uri.host, uri.port, use_ssl: true, open_timeout: 10, read_timeout: 25) do |http|
    http.request(request)
  end
  abort "Data request failed: HTTP #{response.code}" unless response.is_a?(Net::HTTPSuccess)
  JSON.parse(response.body).fetch('entities')
end

countries = %w[Q884 Q17 Q148 Q142 Q183 Q38 Q29 Q145 Q30 Q16 Q408 Q664]
records = entities(countries, 'labels|claims')
capital_ids = countries.map do |country|
  statements = records.fetch(country).fetch('claims').fetch('P36').reject do |statement|
    statement['rank'] == 'deprecated' || statement.fetch('qualifiers', {}).key?('P582')
  end
  preferred = statements.select { |statement| statement['rank'] == 'preferred' }
  current = preferred.empty? ? statements : preferred
  ids = current.map { |statement| statement.dig('mainsnak', 'datavalue', 'value', 'id') }.compact.uniq
  abort "Expected one capital for #{country}, found #{ids.inspect}" unless ids.size == 1
  ids.first
end
capitals = entities(capital_ids.uniq, 'labels')
languages = %w[ko ja en]
facts = languages.to_h do |language|
  pairs = countries.zip(capital_ids).map do |country, capital|
    [records.fetch(country).fetch('labels').fetch(language).fetch('value'),
     capitals.fetch(capital).fetch('labels').fetch(language).fetch('value')]
  end
  [language, pairs]
end
path = ARGV.fetch(0, 'capitals.json')
File.write(path, JSON.pretty_generate(facts) + "\n")
puts "Downloaded #{countries.size} facts in #{languages.size} languages to #{path}"
