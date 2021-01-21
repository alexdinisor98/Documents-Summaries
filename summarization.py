import math
import operator
import re
from collections import Counter, defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge_score import rouge_scorer

from load_dataset import (get_final_dict, get_lemmatizer, get_sentecizer,
                          remove_stop_words)
from my_constants import (BIGRAMS, LEMM_WITH_RM_STOPW, LEMMATIZATION, RAW,
                          RM_STOP_WORDS, UNIGRAMS, docs_test_dir,
                          docs_training_dir, summaries_test_dir,
                          summaries_training_dir)

summarization_classes = ['summary', 'non-summary']


def get_probabilities(training_set_dict, preprocessing_step, N_GRAMS):
    """"Get the prior probability (summary or non-summary) for sentences
    And the likelihood which is the word occurence in a summary sentence.

    Applies Laplace smoothing to solve the zero observations problem.

    :param training_set_dict: Dictionary of the training set.
    :param preprocessing_step: Text preprocessing step.
    :paran N_GRAMS: Unigrams or Bigrams
    :return: Tuple of prior probability and likelihood.
    """

    total_sentences_summary = 0
    total_sentences_non_summary = 0

    # dictionary of ngrams found in summaries or non-summary sentence.
    ngrams_from_ck = defaultdict(list)
    ngram_occ = defaultdict(dict)

    total_words = []
    total_ngrams = []

    for key_class in training_set_dict:
        for key_doc in training_set_dict[key_class]:
            sentenced_orig_doc = get_sentecizer(
                training_set_dict[key_class][key_doc].orig_doc)

            # add space after full stop in summary text
            summary = training_set_dict[key_class][key_doc].summarised_doc
            summarised_text = re.sub(r'\.(?=[^ \W\d])', '. ', summary)

            for sentence in sentenced_orig_doc:
                if preprocessing_step == RM_STOP_WORDS:
                    word_tk = remove_stop_words(word_tokenize(sentence))
                elif preprocessing_step == LEMMATIZATION:
                    word_tk = get_lemmatizer(word_tokenize(sentence))
                elif preprocessing_step == LEMM_WITH_RM_STOPW:
                    word_tk = get_lemmatizer(
                        remove_stop_words(word_tokenize(sentence)))
                else:
                    word_tk = word_tokenize(sentence)

                total_words += word_tk
                if N_GRAMS == BIGRAMS:
                    bigrams_sentence = []
                    for i in range(len(word_tk) - 2):
                        seq = ' '.join(word_tk[i: i + 2])
                        bigrams_sentence.append(seq)

                    total_ngrams += bigrams_sentence
                else:
                    total_ngrams += word_tk

                if sentence in summarised_text:
                    total_sentences_summary += 1
                    if N_GRAMS == BIGRAMS:
                        ngrams_from_ck['summary'] += bigrams_sentence
                    else:
                        ngrams_from_ck['summary'] += word_tk
                else:
                    total_sentences_non_summary += 1
                    if N_GRAMS == BIGRAMS:
                        ngrams_from_ck['non-summary'] += bigrams_sentence
                    else:
                        ngrams_from_ck['non-summary'] += word_tk

    total_ngrams = list(set(total_ngrams))

    for k in summarization_classes:
        ngram_freq_ck = Counter(list(ngrams_from_ck[k]))
        for unique_ngram in ngram_freq_ck.keys():
            ngram_occ[unique_ngram][k] = ngram_freq_ck[unique_ngram]

    total_words = list(set(total_words))
    vocabulary_dim = len(total_words)

    for w in total_ngrams:
        for k in summarization_classes:
            try:
                x = ngram_occ[w][k]
            except KeyError:
                ngram_occ[w][k] = 0

    sentence_probability = defaultdict(dict)
    alpha = 1

    for unique_ngram in ngram_occ:
        for k in ngram_occ[unique_ngram]:
            sentence_probability[unique_ngram][k] = (
                ngram_occ[unique_ngram][k] + alpha) / (len(ngrams_from_ck[k]) + vocabulary_dim + alpha)

    total_sentences = total_sentences_summary + total_sentences_non_summary
    prior_probability = {}
    prior_probability['summary'] = total_sentences_summary / \
        total_sentences
    prior_probability['non-summary'] = total_sentences_non_summary / \
        total_sentences

    return (prior_probability, sentence_probability)


def predict_summarization_class(sentence, prior_probability, sentence_probability, preprocessing_step, N_GRAMS):
    """Predict if the sentence is from summary or non-summary.
    Uses the maximum a posteriori (MAP) estimation.

    Maximizes the log likelihood to prevent underflow.

    :param sentence: Sentence to predict if it is from summary or not.
    :param prior_probability: The prior probability.
    :param sentence_probability: The likelihood.
    :param preprocessing_step: Text preprocessing step.
    :paran N_GRAMS: Unigrams or Bigrams

    :return: The predicted class of the sentence (belongs to summary or not).
    """
    summary_predict = defaultdict(dict)
    ngrams_sentence = []

    if preprocessing_step == RM_STOP_WORDS:
        word_tokens = remove_stop_words(word_tokenize(sentence))
    elif preprocessing_step == LEMMATIZATION:
        word_tokens = get_lemmatizer(word_tokenize(sentence))
    elif preprocessing_step == LEMM_WITH_RM_STOPW:
        word_tokens = get_lemmatizer(
            remove_stop_words(word_tokenize(sentence)))
    else:
        word_tokens = word_tokenize(sentence)

    if N_GRAMS == BIGRAMS:
        for i in range(len(word_tokens) - 2):
            seq = ' '.join(word_tokens[i: i + 2])
            ngrams_sentence.append(seq)
    else:
        ngrams_sentence = word_tokens

    for k in summarization_classes:
        log_likelihood = 0
        for ngram in list(set(ngrams_sentence)):
            try:
                log_likelihood += math.log(sentence_probability[ngram][k])
            except KeyError:
                log_likelihood += 0

        summary_predict[k] = math.log(
            prior_probability[k]) + log_likelihood

    return max(summary_predict.items(), key=operator.itemgetter(1))[0]


def predict(test_set_dict, prior_probability, sentence_probability, preprocessing_step, N_GRAMS):
    """Predict summary text for every document in test set.

    :param test_set_dict: Test set dictionary.
    :param prior_probability: The prior probability.
    :param sentence_probability: The likelihood.
    :param preprocessing_step: Text preprocessing step.
    :paran N_GRAMS: Unigrams or Bigrams
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
                        s, prior_probability, sentence_probability, preprocessing_step, N_GRAMS) == 'summary':
                    result_text += s

            predictions[key_class][key_doc] = result_text

    return predictions


def get_rouge_n(test_set_dict, predictions, N_GRAMS):
    """Get precision and recall for summarization in the context of ROUGE-1 and ROUGE-2.

    :param test_set_dict: The set of records to test the model with.
    :param predictions: Summaries predictions for the test set.
    :param N_GRAMS: Summaries predictions for the test set.
    :return: The precision and recall of the model with unigrams and bigrams.
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

            if N_GRAMS == BIGRAMS:
                rouge2_precision += scores['rouge2'][0]
                rouge2_recall += scores['rouge2'][1]
            else:
                rouge1_precision += scores['rouge1'][0]
                rouge1_recall += scores['rouge1'][1]

            total_predictions += 1

    if N_GRAMS == BIGRAMS:
        mean_rouge2_precision = rouge2_precision / total_predictions
        mean_rouge2_recall = rouge2_recall / total_predictions
        print('Rouge2: ' + 'precision=' + str(mean_rouge2_precision) +
              ' , recall=' + str(mean_rouge2_recall))
    else:
        mean_rouge1_precision = rouge1_precision / total_predictions
        mean_rouge1_recall = rouge1_recall / total_predictions
        print('Rouge1: ' + 'precision=' + str(mean_rouge1_precision) +
              ', recall=' + str(mean_rouge1_recall))


training_set_dict = get_final_dict(docs_training_dir, summaries_training_dir)
test_set_dict = get_final_dict(docs_test_dir, summaries_test_dir)

# change last 2 params for different preprocessing steps and N-grams
(summarization_class_probability,
 sentence_probability) = get_probabilities(training_set_dict, RAW, UNIGRAMS)

predictions = predict(test_set_dict, summarization_class_probability,
                      sentence_probability, RAW, UNIGRAMS)
get_rouge_n(test_set_dict, predictions, UNIGRAMS)


(summarization_class_probability,
 sentence_probability) = get_probabilities(training_set_dict, RAW, BIGRAMS)

predictions = predict(test_set_dict, summarization_class_probability,
                      sentence_probability, RAW, BIGRAMS)
get_rouge_n(test_set_dict, predictions,  BIGRAMS)
