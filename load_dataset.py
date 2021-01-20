import math
import operator
import os
import re
from collections import Counter, defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from constants import LEMM_WITH_RM_STOPW, LEMMATIZATION, RAW, RM_STOP_WORDS


class DocumentProcessor(object):
    """The features of the document.
    """

    def __init__(self, orig_doc=None, word_tokenizer=None, summarised_doc=None):
        """Create a document with tokenization and attached summary.

        :param orig_doc: The original document text.
        :param word_tokenizer: The word tokenization of document.
        :param summarised_doc: The related summary of the document.
        """
        self.orig_doc = orig_doc
        self.word_tokenizer = word_tokenizer
        self.summarised_doc = summarised_doc


def get_documents_dict(articles_dir):
    """Read from directory and get dictionary with documents in categories.

    :param articles_dir: Path to articles directory.
    :return: Dictionary of documents.
    """
    documents_dict_data = defaultdict(dict)

    for d in os.scandir(articles_dir):
        if d.is_dir():
            for f in os.listdir(d.path):
                absolute_path = d.path + '/' + f

                with open(absolute_path, encoding="utf8", errors='ignore') as article_file:
                    data = article_file.read()
                    documents_dict_data[d.name][f] = data

    return documents_dict_data


def get_summaries_dict(summaries_dir):
    """Read from directory and get dictionary with summaries in categories.

    :param summaries_dir: Path to summaries directory.
    :return: Dictionary of summaries.
    """
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
    """Get dictionary with documents and related summaries as dataset.

    :param docs_dir: Path to documents directory.
    :param summaries_dir: Path to summaries directory.
    :return: Dictionary of final dataset.
    """
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


def get_sentecizer(document_text):
    """Tokenizing sentences.
    Consider article text with no full stop after title description, 
    which needs to be removed to take the first sentence after it.

    Example:
    'High fuel prices hit BA's profits

    British Airways has blamed high fuel prices for a 40% drop in profits.'

    :param document_text: Text of original document.
    :return: List of sentences from the document.
    """
    sentecizer = sent_tokenize(document_text)
    try:
        sentecizer[0] = sentecizer[0].split('\n\n')[1]
    except IndexError:
        sentecizer[0] = sentecizer[0]

    return sentecizer


def remove_stop_words(document_tokenized):
    """Remove stop words from a document.
    :param document: Document tokenized in words.

    :return: List of words after removal of stop words.
    """
    stop_words = []
    with open("stop_words") as fp:
        lines = fp.readlines()
        stop_words = [l.strip() for l in lines]

    document_tokenized = [w.lower() for w in document_tokenized]
    return [w for w in document_tokenized if not w in stop_words]


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def get_lemmatizer(document_tokenized):
    """Perform word lemmatization.
    :param document_tokenized: Document tokenized in words.

    :return: List of words after lemmatization.
    """
    lemmatizer = WordNetLemmatizer()
    document_tokenized = [w.lower() for w in document_tokenized]
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w))
            for w in document_tokenized]
