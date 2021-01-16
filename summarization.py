import math
import operator
import re
from collections import Counter, defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge_score import rouge_scorer

from constants import (LEMM_WITH_RM_STOPW, LEMMATIZATION, RAW, RM_STOP_WORDS,
                       docs_test_dir, docs_training_dir, summaries_test_dir,
                       summaries_training_dir)
from load_dataset import get_final_dict, get_sentecizer

summarization_classes = ['summary', 'non-summary']


def get_sentence_probability(training_set_dict):
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

    return max(summary_predict.items(), key=operator.itemgetter(1))[0]


def predict(test_set_dict, summarization_class_probability, sentence_probability):
    """Predict summary text for every document in test set.
    :param test_set_dict: Test set dictionary.
    :return: Predicted target values for test set.
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


def get_rouge_n(test_set_dict, predictions):
    """Get precision and recall in the context of ROUGE-N.
    :param test_set_dict: The set of records to test the model with.
    :param predictions: Predictions for the test set.
    :return: The precision of the model.
    """

    rouge1_precision = 0
    rouge2_precision = 0
    rouge1_recall = 0
    rouge2_recall = 0

    total_predictions = 0

    for key_class in test_set_dict:
        for key_doc in test_set_dict[key_class]:
            ref = test_set_dict[key_class][key_doc].summarised_doc
            scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2'], use_stemmer=True)
            scores = scorer.score(predictions[key_class][key_doc], ref)

            rouge1_precision += scores['rouge1'][0]
            rouge2_precision += scores['rouge2'][0]

            rouge1_recall += scores['rouge1'][1]
            rouge2_recall += scores['rouge2'][1]

            total_predictions += 1

    mean_rouge1_precision = rouge1_precision / total_predictions
    mean_rouge1_recall = rouge1_recall / total_predictions

    mean_rouge2_precision = rouge2_precision / total_predictions
    mean_rouge2_recall = rouge2_recall / total_predictions

    print('Rouge1: ' + 'precision=' + str(mean_rouge1_precision) +
          ', recall=' + str(mean_rouge1_recall))
    print('Rouge2: ' + 'precision=' + str(mean_rouge2_precision) +
          ' , recall=' + str(mean_rouge2_recall))


training_set_dict = get_final_dict(docs_training_dir, summaries_training_dir)
test_set_dict = get_final_dict(docs_test_dir, summaries_test_dir)

(summarization_class_probability,
 sentence_probability) = get_sentence_probability(training_set_dict)

predictions = predict(test_set_dict, summarization_class_probability,
                      sentence_probability)
get_rouge_n(test_set_dict, predictions)

# sentenced_document = get_sentecizer(
#     training_set_dict['business']['001.txt'].orig_doc)

# predictions = ''
# for s in sentenced_document:
#     if predict_summarization_class(
#             s, summarization_class_probability, sentence_probability) == 'summary':
#         predictions += s

# print(predictions)
# ref = training_set_dict['business']['001.txt'].summarised_doc
# scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
# scores = scorer.score(predictions, ref)
# print(scores)
