# Documents-Summaries
Naive Bayes implementation for News Classifier and Text Summarization.

Dataset is collected from BBC News containing articles 
from 5 categories (business, entertainment, politics, sport, tech).

Combines some Text Preprocessing Techniques using **nltk**: Word and Sentence Tokenization, Removing Stop Words and Lemmatization of Words. 

Uses the Maximum a posteriori (MAP) estimation.

Maximizes the log likelihood to prevent Underflow.

Applies Laplace Smoothing to solve the Zero Observations Problem.

* News Classifier.

Performance evaluation by computing Precision and Recall of the model.

* Extractive Summarization.

Performance evaluation by computing Precision and Recall in the Context of ROUGE-N (Words N-Grams Model) to measure unigrams (ROUGE-1) and bigrams (ROUGE-2) with **rouge-score 0.0.4**

