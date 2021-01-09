import os
from collections import defaultdict

from nltk.tokenize import sent_tokenize, word_tokenize

word_tokenizer = defaultdict(dict)
sentecizer = defaultdict(dict)
# for key_category in final_dict:
#     for key_doc in final_dict[key_category]:
#         article_doc = final_dict[key_category][key_doc][0]

#         word_tokenizer[key_category][key_doc] = word_tokenize(article_doc)
#         sentecizer[key_category][key_doc] = sent_tokenize(article_doc)
