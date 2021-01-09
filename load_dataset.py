import os
from collections import defaultdict

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


def get_articles_dict():
    articles_dir = 'BBC News Summary/News Articles/'
    articles_dict_data = defaultdict(dict)

    for d in os.scandir(articles_dir):
        if d.is_dir():

            for f in os.listdir(d.path):
                absolute_path = d.path + '/' + f

                with open(absolute_path, encoding="utf8", errors='ignore') as article_file:
                    # print(absolute_path)
                    data = article_file.read()
                    articles_dict_data[d.name][f] = data
                    # final_dict[d.name][f] = data
                    # exit(0)

    # print(articles_dict_data['business']['006.txt'])
    # print('_________________')
    return articles_dict_data


# get_articles_dict()
print('_____________')
print()


def get_summaries_dict():
    summaries_dir = 'BBC News Summary/Summaries/'
    summaries_dict_data = defaultdict(dict)

    for d in os.scandir(summaries_dir):
        if d.is_dir():

            for f in os.listdir(d.path):
                absolute_path = d.path + '/' + f

                with open(absolute_path, encoding="utf8", errors='ignore') as article_file:
                    # print(absolute_path)
                    data = article_file.read()
                    summaries_dict_data[d.name][f] = data
                    # final_dict[d.name][f] = data
                    # exit(0)
    # print(summaries_dict_data['business']['006.txt'])
    # print('_________________')
    return summaries_dict_data


# get_summaries_dict()


def get_final_articles_dict():
    final_dict = defaultdict(dict)
    articles_dict_data = get_articles_dict()
    summaries_dict_data = get_summaries_dict()

    for key_category in articles_dict_data:

        for key_doc in articles_dict_data[key_category]:
            final_dict[key_category][key_doc] = [articles_dict_data[key_category]
                                                 [key_doc], summaries_dict_data[key_category][key_doc]]
    return final_dict


# mystr = get_final_articles_dict()['business']['006.txt'][0]

# tokenizer in words and sentecizer


def get_word_tokenizer():
    word_tokenizer = defaultdict(dict)
    final_dict = get_final_articles_dict()

    for key_category in final_dict:
        for key_doc in final_dict[key_category]:
            article_doc = final_dict[key_category][key_doc][0]
            word_tokenizer[key_category][key_doc] = word_tokenize(article_doc)

    return word_tokenizer


def get_sentecizer():
    sentecizer = defaultdict(dict)
    final_dict = get_final_articles_dict()

    for key_category in final_dict:
        for key_doc in final_dict[key_category]:
            article_doc = final_dict[key_category][key_doc][0]

            sentecizer[key_category][key_doc] = sent_tokenize(article_doc)

    return sentecizer


def remove_stop_words():
    # read and put the stop words in a list
    stop_words = []
    with open("stop_words") as fp:
        lines = fp.readlines()
        stop_words = [l.strip() for l in lines]

    print(stop_words)
    print('_____________')

    # removing stop words with nltk
    removed_stopwords_articles = defaultdict(dict)
    word_tokenizer = get_word_tokenizer()

    for key_category in word_tokenizer:
        for key_doc in word_tokenizer[key_category]:
            # print(word_tokens)
            # print('___________________')
            word_tokens = word_tokenizer[key_category][key_doc]
            removed_stopwords_articles[key_category][key_doc] = [
                w for w in word_tokens if not w in stop_words]

    print(removed_stopwords_articles['business']['006.txt'])
    return removed_stopwords_articles


# remove_stop_words()


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def get_lemmatizer():
    lemmatizer = WordNetLemmatizer()
    word_tokenizer = get_word_tokenizer()
    final_dict = get_final_articles_dict()
    lemmatizer_dict = defaultdict(dict)
    # sentence = "The striped bats are hanging on their feet for best"
    # for key_category in final_dict:
    #     for key_doc in final_dict[key_category]:
    #         article_doc = final_dict[key_category][key_doc][0]
    #         lemmatizer_dict[key_category][key_doc] = [lemmatizer.lemmatize(w, get_wordnet_pos(w))
    #                                                   for w in word_tokenizer[key_category][key_doc]]
    my_list = word_tokenizer['business']['006.txt']
    my_list = [w.lower() for w in my_list]
    print(word_tokenizer['business']['006.txt'])
    print('_________________________')
    print([lemmatizer.lemmatize(w, get_wordnet_pos(w))
           for w in my_list])
    # print(lemmatizer_dict['business']['006.txt'])


get_lemmatizer()
