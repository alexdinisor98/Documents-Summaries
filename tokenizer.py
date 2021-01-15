import os
from collections import Counter, defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

word_tokenizer = defaultdict(dict)
sentecizer = defaultdict(dict)
# for key_category in final_dict:
#     for key_doc in final_dict[key_category]:
#         article_doc = final_dict[key_category][key_doc][0]

#         word_tokenizer[key_category][key_doc] = word_tokenize(article_doc)
#         sentecizer[key_category][key_doc] = sent_tokenize(article_doc)

words_list = ['Ink', 'helps', 'drive', 'democracy', 'in', 'Asia', 'The', 'Kyrgyz', 'Republic', ',', 'a', 'small', ',', 'mountainous', 'state', 'of', 'the', 'former', 'Soviet', 'republic', ',', 'is', 'using', 'invisible', 'ink', 'and', 'ultraviolet', 'readers', 'in', 'the', 'country', "'s", 'elections', 'as', 'part', 'of', 'a', 'drive', 'to', 'prevent', 'multiple', 'voting', '.', 'This', 'new', 'technology', 'is', 'causing', 'both', 'worries', 'and', 'guarded', 'optimism', 'among', 'different', 'sectors', 'of', 'the', 'population', '.', 'In', 'an', 'effort', 'to', 'live', 'up', 'to', 'its', 'reputation', 'in', 'the', '1990s', 'as', '``', 'an', 'island', 'of', 'democracy', "''", ',', 'the', 'Kyrgyz', 'President', ',', 'Askar', 'Akaev', ',', 'pushed', 'through', 'the', 'law', 'requiring', 'the', 'use', 'of', 'ink', 'during', 'the', 'upcoming', 'Parliamentary', 'and', 'Presidential', 'elections', '.', 'The', 'US', 'government', 'agreed', 'to', 'fund', 'all', 'expenses', 'associated', 'with', 'this', 'decision', '.', 'The', 'Kyrgyz', 'Republic', 'is', 'seen', 'by', 'many', 'experts', 'as', 'backsliding', 'from', 'the', 'high', 'point', 'it', 'reached', 'in', 'the', 'mid-1990s', 'with', 'a', 'hastily', 'pushed', 'through', 'referendum', 'in', '2003', ',', 'reducing', 'the', 'legislative', 'branch', 'to', 'one', 'chamber', 'with', '75', 'deputies', '.', 'The', 'use', 'of', 'ink', 'is', 'only', 'one', 'part', 'of', 'a', 'general', 'effort', 'to', 'show', 'commitment', 'towards', 'more', 'open', 'elections', '-', 'the', 'German', 'Embassy', ',', 'the', 'Soros', 'Foundation', 'and', 'the', 'Kyrgyz', 'government', 'have', 'all', 'contributed', 'to', 'purchase', 'transparent', 'ballot', 'boxes', '.', 'The', 'actual', 'technology', 'behind', 'the', 'ink', 'is', 'not', 'that', 'complicated', '.', 'The', 'ink', 'is', 'sprayed', 'on', 'a', 'person', "'s", 'left', 'thumb', '.', 'It', 'dries', 'and', 'is', 'not', 'visible', 'under', 'normal', 'light', '.', 'However', ',', 'the', 'presence', 'of', 'ultraviolet', 'light',
              '(', 'of', 'the', 'kind', 'used', 'to', 'verify', 'money', ')', 'causes', 'the', 'ink', 'to', 'glow', 'with', 'a', 'neon', 'yellow', 'light', '.', 'At', 'the', 'entrance', 'to', 'each', 'polling', 'station', ',', 'one', 'election', 'official', 'will', 'scan', 'voter', "'s", 'fingers', 'with', 'UV', 'lamp', 'before', 'allowing', 'them', 'to', 'enter', ',', 'and', 'every', 'voter', 'will', 'have', 'his/her', 'left', 'thumb', 'sprayed', 'with', 'ink', 'before', 'receiving', 'the', 'ballot', '.', 'If', 'the', 'ink', 'shows', 'under', 'the', 'UV', 'light', 'the', 'voter', 'will', 'not', 'be', 'allowed', 'to', 'enter', 'the', 'polling', 'station', '.', 'Likewise', ',', 'any', 'voter', 'who', 'refuses', 'to', 'be', 'inked', 'will', 'not', 'receive', 'the', 'ballot', '.', 'These', 'elections', 'are', 'assuming', 'even', 'greater', 'significance', 'because', 'of', 'two', 'large', 'factors', '-', 'the', 'upcoming', 'parliamentary', 'elections', 'are', 'a', 'prelude', 'to', 'a', 'potentially', 'regime', 'changing', 'presidential', 'election', 'in', 'the', 'Autumn', 'as', 'well', 'as', 'the', 'echo', 'of', 'recent', 'elections', 'in', 'other', 'former', 'Soviet', 'Republics', ',', 'notably', 'Ukraine', 'and', 'Georgia', '.', 'The', 'use', 'of', 'ink', 'has', 'been', 'controversial', '-', 'especially', 'among', 'groups', 'perceived', 'to', 'be', 'pro-government', '.', 'Widely', 'circulated', 'articles', 'compared', 'the', 'use', 'of', 'ink', 'to', 'the', 'rural', 'practice', 'of', 'marking', 'sheep', '-', 'a', 'still', 'common', 'metaphor', 'in', 'this', 'primarily', 'agricultural', 'society', '.', 'The', 'author', 'of', 'one', 'such', 'article', 'began', 'a', 'petition', 'drive', 'against', 'the', 'use', 'of', 'the', 'ink', '.', 'The', 'greatest', 'part', 'of', 'the', 'opposition', 'to', 'ink', 'has', 'often', 'been', 'sheer', 'ignorance', '.', 'Local', 'newspapers', 'have', 'carried', 'stories', 'that', 'the', 'ink', 'is', 'harmful', ',', 'radioactive', 'or', 'even', 'that', 'the', 'ultraviolet', 'readers', 'may', 'cause', 'health', 'problems', '.', 'Others', ',', 'such', 'as', 'the', 'aggressively', 'middle', 'of', 'the', 'road', ',', 'Coalition', 'of', 'Non-governmental', 'Organizations', ',', 'have', 'lauded', 'the', 'move', 'as', 'an', 'important', 'step', 'forward', '.', 'This', 'type', 'of', 'ink', 'has', 'been', 'used', 'in', 'many', 'elections', 'in', 'the', 'world', ',', 'in', 'countries', 'as', 'varied', 'as', 'Serbia', ',', 'South', 'Africa', ',', 'Indonesia', 'and', 'Turkey', '.', 'The', 'other', 'common', 'type', 'of', 'ink', 'in', 'elections', 'is', 'indelible', 'visible', 'ink', '-', 'but', 'as', 'the', 'elections', 'in', 'Afghanistan', 'showed', ',', 'improper', 'use', 'of', 'this', 'type', 'of', 'ink', 'can', 'cause', 'additional', 'problems', '.', 'The', 'use', 'of', '``', 'invisible', "''", 'ink', 'is', 'not', 'without', 'its', 'own', 'problems', '.', 'In', 'most', 'elections', ',', 'numerous', 'rumors', 'have', 'spread', 'about', 'it', '.', 'In', 'Serbia', ',', 'for', 'example', ',', 'both', 'Christian', 'and', 'Islamic', 'leaders', 'assured', 'their', 'populations', 'that', 'its', 'use', 'was', 'not', 'contrary', 'to', 'religion', '.', 'Other', 'rumours', 'are', 'associated', 'with', 'how', 'to', 'remove', 'the', 'ink', '-', 'various', 'soft', 'drinks', ',', 'solvents', 'and', 'cleaning', 'products', 'are', 'put', 'forward', '.', 'However', ',', 'in', 'reality', ',', 'the', 'ink', 'is', 'very', 'effective', 'at', 'getting', 'under', 'the', 'cuticle', 'of', 'the', 'thumb', 'and', 'difficult', 'to', 'wash', 'off', '.', 'The', 'ink', 'stays', 'on', 'the', 'finger', 'for', 'at', 'least', '72', 'hours', 'and', 'for', 'up', 'to', 'a', 'week', '.', 'The', 'use', 'of', 'ink', 'and', 'readers', 'by', 'itself', 'is', 'not', 'a', 'panacea', 'for', 'election', 'ills', '.', 'The', 'passage', 'of', 'the', 'inking', 'law', 'is', ',', 'nevertheless', ',', 'a', 'clear', 'step', 'forward', 'towards', 'free', 'and', 'fair', 'elections', '.', "''", 'The', 'country', "'s", 'widely', 'watched', 'parliamentary', 'elections', 'are', 'scheduled', 'for', '27', 'February', '.', 'David', 'Mikosz', 'works', 'for', 'the', 'IFES', ',', 'an', 'international', ',', 'non-profit', 'organisation', 'that', 'supports', 'the', 'building', 'of', 'democratic', 'societies', '.']


class ValueClass(object):
    def __init__(self, orig_doc=None, words_list=None):
        self.orig_doc = orig_doc
        self.words_list = words_list

    def get_words_list(self):
        return self.words_list

    def set_words_list(self, x):
        self.words_list = x


final_dict = defaultdict(dict)
final_dict['tech']['001.txt'] = ValueClass(words_list=words_list)
final_dict['tech']['001.txt'].orig_doc = 'abcd'

# print(final_dict['tech']['001.txt'].orig_doc)

test_list1 = ['y', 'ab', '123', 'ab', '123', 'asw']
x1 = Counter(test_list1)
test_list2 = ['z', 'ab', '123', 'ab', '123', 'asw', 'qwer', '123']
x2 = Counter(test_list2)
for k in x2.keys():
    print(str(k) + ' - ' + str(x2[k]))


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# lemmatizer = WordNetLemmatizer()
# mystr = 'A walking stones. abc 4 corpus 5 walks 2 running? Stone walked.'
# mystr = [w.lower() for w in word_tokenize(mystr)]
# print(mystr)
# print([lemmatizer.lemmatize(w, get_wordnet_pos(w))
#        for w in mystr])
test_list1 = ['y', 'ab', '123', 'ab', '123', 'asw']
print(list(set(test_list1)))

l1 = defaultdict(dict)
l1[0] = test_list1
l1[1] = test_list2
print(list(l1.values()))
print(list(set([item for sublist in list(l1.values()) for item in sublist])))
