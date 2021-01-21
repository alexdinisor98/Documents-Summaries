import math
import operator
import re
from collections import Counter, defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics

from load_dataset import (get_final_dict, get_lemmatizer, get_sentecizer,
                          remove_stop_words)
from my_constants import (LEMM_WITH_RM_STOPW, LEMMATIZATION, RAW,
                          RM_STOP_WORDS, docs_test_dir, docs_training_dir,
                          summaries_test_dir, summaries_training_dir)

ck = ['business', 'entertainment', 'politics', 'sport', 'tech']


def get_prior_probability(training_set_dict):
    """Get the prior probability of class.

    :param training_set_dict: Dictionary of the training set.
    :return: The prior probability.
    """
    total_docs = sum(len(training_set_dict[k]) for k in ck)
    return {k: len(training_set_dict[k]) / total_docs for k in ck}


def get_word_probability(training_set_dict, preprocessing_step):
    """Get the likelihood which is the word occurence probability in a document.

    Applies Laplace smoothing to solve the zero observations problem.

    :param training_set_dict: Dictionary of the training set.
    :param preprocessing_step: Text preprocessing step.
    :return: The likelihood probability of every word.
    """

    word_probability = defaultdict(dict)

    alpha = 1
    vocabulary_dim = 0

    total_words_ck = defaultdict(list)
    word_occurence = defaultdict(dict)
    total_words = []

    for key_class in training_set_dict:
        for key_doc in training_set_dict[key_class]:

            if preprocessing_step == RM_STOP_WORDS:
                doc = remove_stop_words(
                    training_set_dict[key_class][key_doc].word_tokenizer)
            elif preprocessing_step == LEMMATIZATION:
                doc = get_lemmatizer(
                    training_set_dict[key_class][key_doc].word_tokenizer)
            elif preprocessing_step == LEMM_WITH_RM_STOPW:
                doc = get_lemmatizer(remove_stop_words(
                    training_set_dict[key_class][key_doc].word_tokenizer))
            else:
                doc = training_set_dict[key_class][key_doc].word_tokenizer

            total_words_ck[key_class] += doc
            total_words += doc

        word_freq_ck = Counter(list(total_words_ck[key_class]))
        for unique_w in word_freq_ck.keys():
            word_occurence[unique_w][key_class] = word_freq_ck[unique_w]

    total_words = list(set(total_words))
    vocabulary_dim = len(total_words)

    # solving KeyError for certain words which do not appear in some classes
    for w in total_words:
        for c in ck:
            try:
                x = word_occurence[w][c]
            except KeyError:
                word_occurence[w][c] = 0

    # compute probability for each word
    for unique_w in word_occurence:
        for c in word_occurence[unique_w]:
            word_probability[unique_w][c] = (
                word_occurence[unique_w][c] + alpha) / (len(total_words_ck[c]) + vocabulary_dim + alpha)

    return word_probability


def predict_class(document, prior_probability, word_probability, preprocessing_step):
    """Predict the class for the document.
    Uses the maximum a posteriori (MAP) estimation.

    Maximizes the log likelihood to prevent Underflow.

    :param document: Document to predict a class for.
    :param prior_probability: The prior probability.
    :param word_probability: The likelihood.
    :param preprocessing_step: Text preprocessing step.
    :return: The predicted class.
    """

    if preprocessing_step == RM_STOP_WORDS:
        doc = remove_stop_words(document.word_tokenizer)
    elif preprocessing_step == LEMMATIZATION:
        doc = get_lemmatizer(document.word_tokenizer)
    elif preprocessing_step == LEMM_WITH_RM_STOPW:
        doc = get_lemmatizer(remove_stop_words(document.word_tokenizer))
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

        doc_predict_sum[c] = math.log(prior_probability[c]) + log_likelihood

    return max(doc_predict_sum.items(), key=operator.itemgetter(1))[0]


def predict(test_set_dict, prior_probability, word_probability, preprocessing_step):
    """Predict target class for documents in test set.

    :param test_set_dict: Test set dictionary.
    :param prior_probability: The prior probability.
    :param word_probability: The likelihood.
    :param preprocessing_step: Text preprocessing step.
    :return: Predicted target values for test set.
    """

    predictions = defaultdict(dict)
    for key_class in test_set_dict:
        for key_doc in test_set_dict[key_class]:
            result = predict_class(
                test_set_dict[key_class][key_doc], prior_probability, word_probability, preprocessing_step)
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
    """Get recall of predictions on test set.

    :param test_set_dict: The set of records to test the model with.
    :param predictions: Predictions for the test set.
    :param article_class: Class ck (news category).
    :return: The recall of the class model.
    """

    num_correct_ck = 0
    num_documents_ck = len(test_set_dict[article_class].values())

    for key_doc in test_set_dict[article_class]:
        if article_class == predictions[article_class][key_doc]:
            num_correct_ck += 1

    return num_correct_ck / float(num_documents_ck)


training_set_dict = get_final_dict(docs_training_dir, summaries_training_dir)
test_set_dict = get_final_dict(docs_test_dir, summaries_test_dir)

prior_probability = get_prior_probability(training_set_dict)

# raw
print('---- RAW ----')
word_probability = get_word_probability(training_set_dict, RAW)

raw_precision = get_precision(test_set_dict, predict(
    test_set_dict, prior_probability, word_probability, RAW))
print(raw_precision)

raw_recall = {c: get_recall(
    test_set_dict, predict(test_set_dict, prior_probability, word_probability, RAW), c) for c in ck}
print(raw_recall)

# removing stop words
# print('---- RM STOP WORDS ----')
# word_probability = get_word_probability(training_set_dict, RM_STOP_WORDS)

# rm_stop_words_precision = get_precision(test_set_dict, predict(
#     test_set_dict, prior_probability, word_probability, RM_STOP_WORDS))
# print(rm_stop_words_precision)
# rm_stop_words_recall = {c: get_recall(
#     test_set_dict, predict(test_set_dict, prior_probability, word_probability, RM_STOP_WORDS), c) for c in ck}
# print(rm_stop_words_recall)


# # with lemmatization of words
# print()
# print('---- LEMMATIZATION ----')
# word_prob = get_word_probability(training_set_dict, LEMMATIZATION)

# lemm_with_rm_stopw_precision = get_precision(test_set_dict, predict(
#     test_set_dict, prior_probability, word_prob, LEMMATIZATION))
# print(lemm_with_rm_stopw_precision)

# lemm_with_rm_stopw_recall = {c: get_recall(
#     test_set_dict, predict(test_set_dict, prior_probability, word_prob, LEMMATIZATION), c) for c in ck}
# print(lemm_with_rm_stopw_recall)


# # with lemmatization of words with stop words removal
# print()
# print('---- LEMM WITH RM STOP WORDS ----')
# word_prob = get_word_probability(training_set_dict, LEMM_WITH_RM_STOPW)

# lemm_with_rm_stopw_precision = get_precision(test_set_dict, predict(
#     test_set_dict, prior_probability, word_prob, LEMM_WITH_RM_STOPW))
# print(lemm_with_rm_stopw_precision)

# lemm_with_rm_stopw_recall = {c: get_recall(
#     test_set_dict, predict(test_set_dict, prior_probability, word_prob, LEMM_WITH_RM_STOPW), c) for c in ck}
# print(lemm_with_rm_stopw_recall)

# # Predicted values
# y_pred_business = predict(test_set_dict, prior_probability,
#                           word_probability, RAW)['business']
# # Actual values
# y_act = ['business' for i in test_set_dict['business']]
# # Printing the confusion matrix
# # The columns will show the instances predicted for each label,
# # and the rows will show the actual number of instances for each label.
# print(metrics.confusion_matrix(y_act, y_pred_business, labels=["business"]))
# # Printing the precision and recall, among other metrics
# print(metrics.classification_report(
#     y_act, y_pred_business, labels=["business"]))
