# Documents-Summaries
Naive Bayes implementation for a news classifier (and a document summaries).
Dataset is collected from BBC News containing articles 
from 5 categories (business, entertainment, politics, sport, tech).

Combines some text preprocessing techniques using nltk: Word and Sentence Tokenization, 
Removing Stop Words and Lemmatization of Words. 

Maximizes the log likelihood to prevent underflow,
and applies Laplace smoothing to solve the zero observations problem.

precision = 0.9732620320855615
recall = {'business': 0.9595959595959596, 'entertainment': 0.9243697478991597, 'politics': 1.0, 'sport': 0.9911504424778761, 'tech': 0.990909090909091}

Extractive Summarization using Naive Bayes.