# ============================================================
# NATURAL LANGUAGE PROCESSING (NLP) USING PYTHON
# Author: Sathya V
# ============================================================

import nltk
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = """
Natural Language Processing is a branch of Artificial Intelligence.
It enables computers to understand human language.
"""

print("Original Text:\n", text)

sentences = sent_tokenize(text)
print("\nSentences:", sentences)

words = word_tokenize(text)
print("\nWords:", words)

stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w.lower() not in stop_words]

print("\nFiltered Words:", filtered_words)

stemmer = PorterStemmer()
print("\nStemmed Words:", [stemmer.stem(w) for w in filtered_words])

lemmatizer = WordNetLemmatizer()
print("\nLemmatized Words:", [lemmatizer.lemmatize(w) for w in filtered_words])

review = "The NLP course is excellent and informative."
blob = TextBlob(review)

print("\nSentiment Polarity:", blob.sentiment.polarity)

documents = [
    "Natural Language Processing is interesting",
    "Machine Learning and NLP are related fields",
    "Artificial Intelligence uses NLP techniques"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("\nTF-IDF Features:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())

print("\nNLP Program Completed Successfully")
