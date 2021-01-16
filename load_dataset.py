import math
import operator
import os
import re
from collections import Counter, defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge_score import rouge_scorer

RAW = 0
RM_STOP_WORDS = 1
LEMMATIZATION = 2
LEMM_WITH_RM_STOPW = 3


class DocumentProcessor(object):
    def __init__(self, orig_doc=None, word_tokenizer=None, summarised_doc=None):
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
            word_tokens = [w.lower() for w in word_tokens]

            sentecizer = sent_tokenize(doc)

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


def get_sentecizer(document_text):
    abc = sent_tokenize(document_text)
    try:
        abc[0] = abc[0].split('\n\n')[1]
    except IndexError:
        abc[0] = abc[0]

    return abc


def remove_stop_words(document):

    stop_words = []
    with open("stop_words") as fp:
        lines = fp.readlines()
        stop_words = [l.strip() for l in lines]

    document_tokenized = [w.lower() for w in document.word_tokenizer]
    return [w for w in document.word_tokenizer if not w in stop_words]


# print(remove_stop_words(training_set_dict['business']['001.txt']))

# print('dim voc article -> ' +
#       str(len(training_set_dict['business']['001.txt'].word_tokenizer)))
# print('dim voc RM STOP WORDS article -> ' +
#       str(len(remove_stop_words(training_set_dict['business']['001.txt']))))


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def get_lemmatizer(document_tokenized):
    lemmatizer = WordNetLemmatizer()
    document_tokenized = [w.lower() for w in document_tokenized]
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w))
            for w in document_tokenized]


# (get_lemmatizer(training_dict['business']['001.txt']))


ck = ['business', 'entertainment', 'politics', 'sport', 'tech']

total_docs = sum(len(training_set_dict[k]) for k in ck)
class_probability = {k: len(training_set_dict[k]) / total_docs for k in ck}


def get_word_probability(preprocessing_step):
    word_probability = defaultdict(dict)

    alpha = 1
    vocabulary_dim = 0

    total_words_ck = defaultdict(list)
    word_occurence = defaultdict(dict)
    total_words = []

    for key_class in training_set_dict:
        for key_doc in training_set_dict[key_class]:

            if preprocessing_step == RM_STOP_WORDS:
                doc = remove_stop_words(training_set_dict[key_class][key_doc])
            elif preprocessing_step == LEMMATIZATION:
                doc = get_lemmatizer(
                    training_set_dict[key_class][key_doc].word_tokenizer)
            elif preprocessing_step == LEMM_WITH_RM_STOPW:
                doc = get_lemmatizer(remove_stop_words(
                    training_set_dict[key_class][key_doc]))
            else:
                doc = training_set_dict[key_class][key_doc].word_tokenizer

            total_words_ck[key_class] += doc
            total_words += doc

        word_freq_ck = Counter(list(total_words_ck[key_class]))
        for unique_w in word_freq_ck.keys():
            word_occurence[unique_w][key_class] = word_freq_ck[unique_w]

    print(len(total_words))
    total_words = list(set(total_words))
    vocabulary_dim = len(total_words)
    print('VOCABULARY DIM: ' + str(vocabulary_dim))
    # for c in ck:
    #     total_words_ck[c] = list(set(total_words_ck[c]))

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

    return word_probability


def predict_class(document, class_probability, word_probability, preprocessing_step):
    """Predict the class for the document.

    Maximizes the log likelihood to prevent underflow.

    :param document: Document to predict a class for.
    :return: The predicted class.
    """

    if preprocessing_step == RM_STOP_WORDS:
        doc = remove_stop_words(document)
    elif preprocessing_step == LEMMATIZATION:
        doc = get_lemmatizer(document.word_tokenizer)
    elif preprocessing_step == LEMM_WITH_RM_STOPW:
        doc = get_lemmatizer(remove_stop_words(document))
    else:
        doc = document.word_tokenizer

    doc_predict_sum = defaultdict(dict)
    for c in ck:
        log_likelihood = 0
        for w in list(set(doc)):
            try:
                log_likelihood += math.log(word_probability[w][c])
            except KeyError:
                log_likelihood += 0

        doc_predict_sum[c] = math.log(class_probability[c]) + log_likelihood

    return max(doc_predict_sum.items(), key=operator.itemgetter(1))[0]


def predict(test_set_dict, class_probability, word_probability, preprocessing_step):
    """Predict target values for test set.
    :param test_set_dict: Test set dictionary.
    :return: Predicted target values for test set.
    """

    predictions = defaultdict(dict)
    for key_class in test_set_dict:
        for key_doc in test_set_dict[key_class]:
            result = predict_class(
                test_set_dict[key_class][key_doc], class_probability, word_probability, preprocessing_step)
            predictions[key_class][key_doc] = result

    return predictions


def get_precision(test_set_dict, predictions):
    """Get precision of predictions on test set.
    :param test_set_dict: The set of records to test the model with.
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


def get_recall(test_set_dict, predictions, article_class):
    """Get recall of predictions on test set .
    :param test_set_dict: The set of records to test the model with.
    :param predictions: Predictions for the test set.
    :param article_class: Class ck.
    :return: The recall of the class model.
    """

    num_correct_ck = 0

    num_documents_ck = len(test_set_dict[article_class].values())

    for key_doc in test_set_dict[article_class]:
        if article_class == predictions[article_class][key_doc]:
            num_correct_ck += 1

    return num_correct_ck / float(num_documents_ck)


# word_probability = get_word_probability(RM_STOP_WORDS)
# print(predict_class(
#     training_set_dict['business']['001.txt'], class_probability, word_probability, RM_STOP_WORDS))
# print(predict_class(training_set_dict['entertainment']
#                     ['001.txt'], class_probability, word_probability, RM_STOP_WORDS))

# # raw
# print('---- RAW ----')
# word_probability = get_word_probability(RAW)

# raw_precision = get_precision(test_set_dict, predict(
#     test_set_dict, class_probability, word_probability, RAW))
# print(raw_precision)

# raw_recall = {c: get_recall(
#     test_set_dict, predict(test_set_dict, class_probability, word_probability, RAW), c) for c in ck}
# print(raw_recall)

# # removing stop words
# print('---- RM STOP WORDS ----')
# word_probability = get_word_probability(RM_STOP_WORDS)

# rm_stop_words_precision = get_precision(test_set_dict, predict(
#     test_set_dict, class_probability, word_probability, RM_STOP_WORDS))
# print(rm_stop_words_precision)
# rm_stop_words_recall = {c: get_recall(
#     test_set_dict, predict(test_set_dict, class_probability, word_probability, RM_STOP_WORDS), c) for c in ck}
# print(rm_stop_words_recall)

# # with lemmatization of words
# print()
# print('---- LEMM WITH RM STOP WORDS ----')
# word_prob = get_word_probability(LEMM_WITH_RM_STOPW)

# lemm_with_rm_stopw_precision = get_precision(test_set_dict, predict(
#     test_set_dict, class_probability, word_prob, LEMM_WITH_RM_STOPW))
# print(lemm_with_rm_stopw_precision)

# lemm_with_rm_stopw_recall = {c: get_recall(
#     test_set_dict, predict(test_set_dict, class_probability, word_prob, LEMM_WITH_RM_STOPW), c) for c in ck}
# print(lemm_with_rm_stopw_recall)


# orig_document = get_sentecizer(
#     training_set_dict['business']['001.txt'].orig_doc)
# print(len(orig_document))
# print('___________________')
# my_text = training_set_dict['business']['001.txt'].summarised_doc
# summarised_text = re.sub(r'\.(?=[^ \W\d])', '. ', my_text)
# print(get_sentecizer(summarised_text))

summarization_classes = ['summary', 'non-summary']


def get_sentence_probability():
    total_sentences_summary = 0
    total_sentences_non_summary = 0
    words_from_ck = defaultdict(list)
    word_occ = defaultdict(dict)

    total_words = []

    for key_class in training_set_dict:
        for key_doc in training_set_dict[key_class]:
            orig_doc_sentenced = get_sentecizer(
                training_set_dict[key_class][key_doc].orig_doc)

            summary = training_set_dict[key_class][key_doc].summarised_doc
            summarised_text = re.sub(r'\.(?=[^ \W\d])', '. ', summary)

            total_words += training_set_dict[key_class][key_doc].word_tokenizer

            for sentence in orig_doc_sentenced:
                if sentence in summarised_text:
                    total_sentences_summary += 1

                    words_from_ck['summary'] += word_tokenize(sentence)
                else:
                    total_sentences_non_summary += 1
                    words_from_ck['non-summary'] += word_tokenize(sentence)

    print(total_sentences_summary)
    print(total_sentences_non_summary)

    for k in summarization_classes:
        word_freq_ck = Counter(list(words_from_ck[k]))
        for unique_w in word_freq_ck.keys():
            word_occ[unique_w][k] = word_freq_ck[unique_w]

    total_words = list(set(total_words))
    vocabulary_dim = len(total_words)

    for w in total_words:
        for k in summarization_classes:
            try:
                x = word_occ[w][k]
            except KeyError:
                word_occ[w][k] = 0

    sentence_probability = defaultdict(dict)
    alpha = 1

    for unique_w in word_occ:
        for k in word_occ[unique_w]:
            sentence_probability[unique_w][k] = (
                word_occ[unique_w][k] + alpha) / (len(words_from_ck[k]) + vocabulary_dim + alpha)

    total_sentences = total_sentences_summary + total_sentences_non_summary
    summarization_class_probability = {}
    summarization_class_probability['summary'] = total_sentences_summary / \
        total_sentences
    summarization_class_probability['non-summary'] = total_sentences_non_summary / \
        total_sentences

    return (summarization_class_probability, sentence_probability)


def predict_summarization_class(sentence, summarization_class_probability, sentence_probability):
    """Predict if the sentence is summary or non-summary.

    Maximizes the log likelihood to prevent underflow.

    :param document: Document to predict a class for.
    :return: The predicted class.
    """
    summary_predict = defaultdict(dict)

    for k in summarization_classes:
        log_likelihood = 0
        for w in list(set(word_tokenize(sentence))):
            try:
                log_likelihood += math.log(sentence_probability[w][k])
            except KeyError:
                log_likelihood += 0

        summary_predict[k] = math.log(
            summarization_class_probability[k]) + log_likelihood

    print('_____________________________')
    print(sentence)
    return max(summary_predict.items(), key=operator.itemgetter(1))[0]
    # if c_MAP == 'summary':


(summarization_class_probability, sentence_probability) = get_sentence_probability()

sentenced_document = get_sentecizer(
    training_set_dict['business']['001.txt'].orig_doc)

predictions = ''
for s in sentenced_document:
    if predict_summarization_class(
            s, summarization_class_probability, sentence_probability) == 'summary':
        predictions += s

print(predictions)
ref = training_set_dict['business']['001.txt'].summarised_doc
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
scores = scorer.score(predictions, ref)
print(scores)


def predict(test_set_dict, summarization_class_probability, sentence_probability):
    """Predict target values for test set.
    :param test_set_dict: Test set dictionary.
    :return: Predicted target values for test set .
    """

    predictions = defaultdict(dict)
    for key_class in test_set_dict:
        for key_doc in test_set_dict[key_class]:
            sentenced_doc = get_sentecizer(
                test_set_dict[key_class][key_doc].orig_doc)

            result_text = ''
            for s in sentenced_doc:
                if predict_summarization_class(
                        s, summarization_class_probability, sentence_probability) == 'summary':
                    result_text += s

            predictions[key_class][key_doc] = result_text

    return predictions
