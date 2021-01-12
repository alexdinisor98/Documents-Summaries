import math
import operator
import os
from collections import Counter, defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


class DocumentProcessor(object):
    def __init__(self, orig_doc=None, word_tokenizer=None,  summarised_doc=None):
        self.orig_doc = orig_doc
        self.word_tokenizer = word_tokenizer
        self.summarised_doc = summarised_doc


def get_documents_dict(articles_dir):

    documents_dict_data = defaultdict(dict)

    for d in os.scandir(articles_dir):
        if d.is_dir():

            for f in os.listdir(d.path):
                absolute_path = d.path + '/' + f

                with open(absolute_path, encoding="utf8", errors='ignore') as article_file:
                    # print(absolute_path)
                    data = article_file.read()
                    documents_dict_data[d.name][f] = data
                    # final_dict[d.name][f] = data
                    # exit(0)
    return documents_dict_data


def get_summaries_dict(summaries_dir):

    summaries_dict_data = defaultdict(dict)

    for d in os.scandir(summaries_dir):
        if d.is_dir():

            for f in os.listdir(d.path):
                absolute_path = d.path + '/' + f

                with open(absolute_path, encoding="utf8", errors='ignore') as article_file:

                    data = article_file.read()
                    summaries_dict_data[d.name][f] = data

    return summaries_dict_data


def get_final_dict(docs_dir, summaries_dir):
    set_dict = defaultdict(dict)

    docs_dict = get_documents_dict(docs_dir)

    summaries_dict = get_summaries_dict(summaries_dir)

    for key_class in docs_dict:
        for key_doc in docs_dict[key_class]:
            doc = docs_dict[key_class][key_doc]
            word_tokens = word_tokenize(doc)

            set_dict[key_class][key_doc] = DocumentProcessor(
                orig_doc=doc,
                word_tokenizer=word_tokens,
                summarised_doc=summaries_dict[key_class][key_doc])

    return set_dict


docs_training_dir = 'Training Set/News Articles/'
summaries_training_dir = 'Training Set/Summaries/'
training_set_dict = get_final_dict(docs_training_dir, summaries_training_dir)

docs_test_dir = 'Test Set/News Articles/'
summaries_test_dir = 'Test Set/Summaries/'
test_set_dict = get_final_dict(docs_test_dir, summaries_test_dir)
# tokenizer in words and sentecizer


def get_sentecizer():
    sentecizer = defaultdict(dict)

    for key_category in training_set_dict:
        for key_doc in training_set_dict[key_category]:
            article_doc = training_set_dict[key_category][key_doc].orig_doc

            sentecizer[key_category][key_doc] = sent_tokenize(article_doc)

    return sentecizer


def remove_stop_words(document):

    stop_words = []
    with open("stop_words") as fp:
        lines = fp.readlines()
        stop_words = [l.strip() for l in lines]

    return [w for w in document.word_tokenizer if not w in stop_words]


print(len(training_set_dict['business']['001.txt'].word_tokenizer))
print(len(remove_stop_words(training_set_dict['business']['001.txt'])))


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def get_lemmatizer(document):
    lemmatizer = WordNetLemmatizer()
    document.word_tokenizer = [w.lower() for w in document.word_tokenizer]
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w))
            for w in document.word_tokenizer]


# print(get_lemmatizer(training_dict['business']['001.txt']))


ck = ['business', 'entertainment', 'politics', 'sport', 'tech']

total_docs = sum(len(training_set_dict[k]) for k in ck)

class_probability = {k: len(training_set_dict[k]) / total_docs for k in ck}

alpha = 1
vocabulary_dim = 0

total_words_ck = defaultdict(list)
word_probability = defaultdict(dict)


word_occurence = defaultdict(dict)
total_words = []

for key_class in training_set_dict:
    for key_doc in training_set_dict[key_class]:
        total_words_ck[key_class] += training_set_dict[key_class][key_doc].word_tokenizer

    total_words += list(set(total_words_ck[key_class]))

    word_freq_ck = Counter(list(total_words_ck[key_class]))
    for unique_w in word_freq_ck.keys():
        word_occurence[unique_w][key_class] = word_freq_ck[unique_w]

vocabulary_dim = len(total_words)

# solving KeyError for certain words in certain classes


for w in total_words:
    for c in ck:
        try:
            x = word_occurence[w][c]
        except KeyError:
            word_occurence[w][c] = 0


for unique_w in word_occurence:
    for c in word_occurence[unique_w]:
        word_probability[unique_w][c] = (
            word_occurence[unique_w][c] + alpha) / (len(total_words_ck[c]) + vocabulary_dim + alpha)


class_prediction = defaultdict(dict)


def predict_class(document):

    doc_predict_sum = defaultdict(dict)
    for c in ck:
        log_likelihood = 0
        for w in list(set(document.word_tokenizer)):
            try:
                log_likelihood += math.log(word_probability[w][c])
            except KeyError:
                log_likelihood += 0
                # print(w + ' - ' + c)

        doc_predict_sum[c] = class_probability[c] + log_likelihood

    return max(doc_predict_sum.items(), key=operator.itemgetter(1))[0]


print(predict_class(training_set_dict['business']['001.txt']))
print(predict_class(training_set_dict['entertainment']['001.txt']))


def predict(test_set_dict):
    """Predict target values for test set.
    :param test_set: Test set with dimension m x n,
                     where m is the number of examples,
                     and n is the number of features.
    :return: Predicted target values for test set with dimension m,
             where m is the number of examples.
    """

    predictions = defaultdict(dict)
    for key_class in test_set_dict:
        for key_doc in test_set_dict[key_class]:
            result = predict_class(test_set_dict[key_class][key_doc])
            predictions[key_class][key_doc] = result

    return predictions


def get_precision(test_set_dict, predictions):
    """Get precision of predictions on test set.
    :param test_set: The set of records to test the model with.
    :param predictions: Predictions for the test set.
    :return: The precision of the model.
    """
    num_correct = 0

    num_predictions = sum(len(test_set_dict[c].values()) for c in ck)

    for key_class in test_set_dict:
        for key_doc in test_set_dict[key_class]:
            if key_class == predictions[key_class][key_doc]:
                num_correct += 1

    return num_correct / num_predictions


print(get_precision(test_set_dict, predict(test_set_dict)))
