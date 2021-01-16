# Documents-Summaries
Naive Bayes implementation for a news classifier (and a document summaries).
Dataset is collected from BBC News containing articles 
from 5 categories (business, entertainment, politics, sport, tech).

Combines some text preprocessing techniques using nltk: Word and Sentence Tokenization, 
Removing Stop Words and Lemmatization of Words. 

Maximizes the log likelihood to prevent underflow,
and applies Laplace smoothing to solve the zero observations problem.

---- RAW ----
721990
VOCABULARY DIM: 30244
0.9766606822262118
{'business': 0.96875, 'entertainment': 0.9381443298969072, 'politics': 1.0, 'sport': 0.984375, 'tech': 0.99}

---- RM STOP WORDS ----
460122
VOCABULARY DIM: 30124
0.9748653500897666
{'business': 0.9609375, 'entertainment': 0.9381443298969072, 'politics': 1.0, 'sport': 0.984375, 'tech': 0.99}

---- LEMM WITH RM STOP WORDS ----
460122
VOCABULARY DIM: 24886
0.9802513464991023
{'business': 0.9765625, 'entertainment': 0.9484536082474226, 'politics': 1.0, 'sport': 0.984375, 'tech': 0.99}

Extractive Summarization using Naive Bayes.