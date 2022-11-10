
from collections import defaultdict
from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords

from sample_data import response, query, key_fields,field_score


def calculate_cosine_similarity(X,Y):
  if not len(X) or not len(Y):
    return 0
  numerator = 0
  denominator = 0
  for key in X:
    numerator += X[key]*Y[key]
  sum_sqrt_X = 0
  for _,v in X.items():
    sum_sqrt_X += v*v
  sum_sqrt_X = sum_sqrt_X ** 0.5
  sum_sqrt_Y = 0
  for k,v in Y.items():
    sum_sqrt_Y += v*v
  sum_sqrt_Y = sum_sqrt_Y ** 0.5
  denominator = sum_sqrt_X * sum_sqrt_Y
  return numerator/denominator


def get_tokens(document):
  reg_exp = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
  for word in document:
    if not word or word in reg_exp:
      try:
        while True:
          document.remove(word)
      except ValueError:
        pass
  document=[word.lower() for word in document]					
  text_tokens = list()
  for word in document:
    token = word_tokenize(word)
    text_tokens.extend(token)
  return list(set(text_tokens))

def get_response_meta_data(response):
  response_meta_data = []
  for row in response:
    current_meta_data = {}
    current_meta_data["id"] = row["id"]
    for field in key_fields:
      data = row[field]
      if not data:
        continue
      if field not in ["tags", "keywords"]:
        data = list(set(data.split(" ")))
      tokens = get_tokens(data)
      current_meta_data[field] = tokens
    response_meta_data.append(current_meta_data)
  return response_meta_data


def get_results(query, response):
  results_score_map = {}
  response_meta_data = get_response_meta_data(response)
  print(f"Response meta data = {response_meta_data}")
  query_meta_data = get_tokens(query)
  print(f"Query meta data = {query_meta_data}")

  query_score_data = {}
  for field in query_meta_data:
    query_score_data[field] = 1
  print(f"Query weight data = {query_score_data}")

  
  for i,meta_data in enumerate(response_meta_data):
    print(f"\nRow {i+1};#{meta_data['id']}")
    matched_fields = defaultdict(int)
    for field in key_fields:
      if field not in meta_data:
        continue
      current_matched_fields = set(meta_data[field]).intersection(set(query_meta_data))
      for value in current_matched_fields:
        matched_fields[value] += field_score[field]
      print(f"{field} = {current_matched_fields}")
    print(f"Terms weight = {matched_fields}")
    score = calculate_cosine_similarity( query_score_data, matched_fields)
    print(f"Cosine similarity(query,doc) = {score}")
    results_score_map[meta_data["id"]] = score
  results_score_map = {k: v for k, v in sorted(results_score_map.items(), key=lambda item: item[1], reverse=True)}
  return results_score_map

query = query.split(" ")
print(get_results(query, response))

# def get_stem_map(text_tokens):
#   ps = PorterStemmer()
#   stem_map = {}
#   for word in text_tokens:
#     stem_map[word] = ps.stem(word)
#   return stem_map

# def get_tf_idf_map(stem_map, text_tokens, response_data):
#   stemmed_words = [ stem_map[word] for word in text_tokens ]
#   tokens_without_stop_words = [ word for word in stemmed_words if not word in stopwords.words() ]
#   print(f"Token without stop words : {tokens_without_stop_words}")
