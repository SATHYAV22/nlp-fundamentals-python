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
import os
import re
import string
import math
import collections
from collections import Counter
import numpy as np

# ==============================================================================
# 1. TEXT PREPROCESSING & CLEANING MODULE
# ==============================================================================

class TextPreprocessor:
    """
    A comprehensive text preprocessing pipeline handling cleaning, tokenization,
    stopword removal, and basic lemmatization/stemming heuristics.
    """
    def __init__(self, extra_stopwords=None):
        # Standard English stopwords list
        self.stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
            'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
            'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
            're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
            'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
            'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
            "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
            "weren't", 'won', "won't", 'wouldn', "wouldn't"
        }
        if extra_stopwords:
            self.stopwords.update(extra_stopwords)
            
    def clean_text(self, text):
        """Removes HTML tags, URLs, special characters, and forces lowercase."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)  # HTML tags
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # URLs
        text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    def tokenize(self, text):
        """Splits text into individual tokens (words)."""
        return text.split()

    def remove_stopwords(self, tokens):
        """Filters out common stop words from a list of tokens."""
        return [token for token in tokens if token not in self.stopwords]

    def simple_stemmer(self, tokens):
        """Applies basic suffix stripping rules (heuristic stemming)."""
        stemmed = []
        for token in tokens:
            if len(token) > 4:
                if token.endswith('ing'):
                    token = token[:-3]
                elif token.endswith('ly'):
                    token = token[:-2]
                elif token.endswith('es'):
                    token = token[:-2]
                elif token.endswith('s') and not token.endswith('ss'):
                    token = token[:-1]
                elif token.endswith('ed'):
                    token = token[:-2]
            stemmed.append(token)
        return stemmed

    def pipeline(self, text):
        """Executes the end-to-end preprocessing workflow."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        stemmed = self.simple_stemmer(filtered)
        return stemmed


# ==============================================================================
# 2. FEATURE EXTRACTION & VECTORIZATION (TF-IDF FROM SCRATCH)
# ==============================================================================

class TFIDFVectorizer:
    """
    Computes Term Frequency-Inverse Document Frequency matrices for a corpus.
    """
    def __init__(self):
        self.vocabulary = {}
        self.idf_dict = {}
        self.num_documents = 0

    def fit(self, tokenized_corpus):
        """Learns vocabulary and calculates IDF scores from a tokenized corpus."""
        self.num_documents = len(tokenized_corpus)
        unique_words = set(word for doc in tokenized_corpus for word in doc)
        
        # Build vocabulary indices
        self.vocabulary = {word: i for i, word in enumerate(sorted(unique_words))}
        
        # Calculate Document Frequency (DF)
        df_dict = Counter()
        for doc in tokenized_corpus:
            unique_doc_words = set(doc)
            for word in unique_doc_words:
                df_dict[word] += 1
                
        # Calculate Inverse Document Frequency (IDF) with smoothing
        for word, df in df_dict.items():
            self.idf_dict[word] = math.log((1 + self.num_documents) / (1 + df)) + 1

    def transform(self, tokenized_corpus):
        """Transforms tokenized documents into a dense TF-IDF matrix."""
        matrix = np.zeros((len(tokenized_corpus), len(self.vocabulary)))
        
        for doc_idx, doc in enumerate(tokenized_corpus):
            if not doc:
                continue
            word_counts = Counter(doc)
            total_words = len(doc)
            
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    tf = count / total_words
                    idf = self.idf_dict.get(word, 0)
                    matrix[doc_idx, word_idx] = tf * idf
                    
        return matrix

    def fit_transform(self, tokenized_corpus):
        """Fits vocabulary and transitions corpus to TF-IDF representations."""
        self.fit(tokenized_corpus)
        return self.transform(tokenized_corpus)


# ==============================================================================
# 3. N-GRAM LANGUAGE MODEL & TEXT GENERATION
# ==============================================================================

class NGramLanguageModel:
    """
    A statistical N-Gram Language Model using Maximum Likelihood Estimation 
    with Laplace (add-one) smoothing for text generation and perplexity calculation.
    """
    def __init__(self, n=2):
        self.n = n
        self.ngrams = collections.defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()

    def _get_context_and_target(self, tokens):
        """Helper to break a token stream into contexts and target words."""
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            context = ngram[:-1]
            target = ngram[-1]
            yield context, target

    def train(self, tokenized_corpus):
        """Populates frequency tables from a tokenized dataset."""
        for tokens in tokenized_corpus:
            # Pad tokens to handle starts and ends of sequences
            padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
            self.vocab.update(padded_tokens)
            
            for context, target in self._get_context_and_target(padded_tokens):
                self.ngrams[context][target] += 1
                self.context_counts[context] += 1

    def get_probability(self, context, target):
        """Calculates smoothed conditional probability: P(target | context)."""
        context = tuple(context[-(self.n - 1):]) if self.n > 1 else ()
        vocab_size = len(self.vocab)
        
        count_context = self.context_counts[context]
        count_ngram = self.ngrams[context][target]
        
        # Laplace Smoothing
        return (count_ngram + 1) / (count_context + vocab_size)

    def generate_text(self, seed_text, max_words=20):
        """Generates sequence paths using probabilistic sampling."""
        preprocessor = TextPreprocessor()
        tokens = preprocessor.tokenize(seed_text.lower())
        
        context = ['<s>'] * (self.n - 1) + tokens
        context = context[-(self.n - 1):] if self.n > 1 else []
        
        generated = list(tokens)
        vocab_list = list(self.vocab)
        
        for _ in range(max_words):
            ctx_tuple = tuple(context)
            probabilities = [self.get_probability(ctx_tuple, word) for word in vocab_list]
            
            # Normalize probabilities to avoid floating point precision edge cases
            prob_sum = sum(probabilities)
            probabilities = [p / prob_sum for p in probabilities]
            
            next_word = np.random.choice(vocab_list, p=probabilities)
            if next_word == '</s>':
                break
                
            generated.append(next_word)
            if self.n > 1:
                context = context[1:] + [next_word]
                
        return " ".join(generated)


# ==============================================================================
# 4. WORD EMBEDDINGS & SIMILARITY MODULE
# ==============================================================================

class WordEmbeddingSpace:
    """
    Manages a structural map of mock embeddings to illustrate geometric
    semantic relations, cosine similarity, and analogy task workflows.
    """
    def __init__(self):
        self.embeddings = {}
        self.vector_dim = 100

    def generate_mock_embeddings(self, vocab):
        """Seeds deterministic mock dense vectors for structural visualization."""
        np.random.seed(42)
        for word in vocab:
            vec = np.random.uniform(-1.0, 1.0, self.vector_dim)
            self.embeddings[word] = vec / np.linalg.norm(vec)

    def cosine_similarity(self, vec_a, vec_b):
        """Computes the cosine angle similarity metric between two vectors."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def find_most_similar(self, word, top_n=3):
        """Extracts the top N geographically closest words to a target entry."""
        if word not in self.embeddings:
            return []
        
        target_vec = self.embeddings[word]
        similarities = []
        
        for other_word, vec in self.embeddings.items():
            if other_word == word:
                continue
            sim = self.cosine_similarity(target_vec, vec)
            similarities.append((other_word, sim))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def solve_analogy(self, a, b, c, top_n=1):
        """Solves vector arithmetic analogies: Vector(b) - Vector(a) + Vector(c) = Vector(d)"""
        for w in (a, b, c):
            if w not in self.embeddings:
                return f"Word '{w}' not in vocabulary embedding index."
                
        vec_a = self.embeddings[a]
        vec_b = self.embeddings[b]
        vec_c = self.embeddings[c]
        
        target_vec = vec_b - vec_a + vec_c
        target_vec /= np.linalg.norm(target_vec)
        
        results = []
        for word, vec in self.embeddings.items():
            if word in (a, b, c):
                continue
            sim = self.cosine_similarity(target_vec, vec)
            results.append((word, sim))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]


# ==============================================================================
# 5. MACHINE LEARNING ENGINE (NAIVE BAYES CLASSIFIER FOR SENTIMENT ANALYSIS)
# ==============================================================================

class NaiveBayesClassifier:
    """
    A Multinomial Naive Bayes text classifier built entirely from scratch.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_log_prior = {}
        self.feature_log_prob = collections.defaultdict(dict)
        self.classes = []
        self.vocab = set()

    def train(self, X_tokenized, y_labels):
        """Trains the classifier using token frequencies and document counts."""
        n_docs = len(X_tokenized)
        self.classes = list(set(y_labels))
        class_counts = Counter(y_labels)
        
        # Calculate priors
        for c in self.classes:
            self.class_log_prior[c] = math.log(class_counts[c] / n_docs)
            
        # Count words per class
        word_counts_per_class = {c: Counter() for c in self.classes}
        total_words_per_class = {c: 0 for c in self.classes}
        
        for tokens, label in zip(X_tokenized, y_labels):
            for token in tokens:
                self.vocab.add(token)
                word_counts_per_class[label][token] += 1
                total_words_per_class[label] += 1
                
        # Calculate conditional word log probabilities
        vocab_size = len(self.vocab)
        for c in self.classes:
            for token in self.vocab:
                count = word_counts_per_class[c][token]
                # Log likelihood computation with smoothing
                log_prob = math.log((count + self.alpha) / (total_words_per_class[c] + self.alpha * vocab_size))
                self.feature_log_prob[c][token] = log_prob

    def predict_single(self, tokenized_text):
        """Predicts the target class label for a single document list."""
        posteriors = {}
        
        for c in self.classes:
            log_posterior = self.class_log_prior[c]
            for token in tokenized_text:
                if token in self.vocab:
                    log_posterior += self.feature_log_prob[c][token]
            posteriors[c] = log_posterior
            
        # Return class maximizing log posterior likelihood
        return max(posteriors, key=posteriors.get)

    def predict(self, X_tokenized):
        """Batch predicts evaluation rows across multiple document listings."""
        return [self.predict_single(doc) for doc in X_tokenized]


# ==============================================================================
# 6. PIPELINE ORCHESTRATION & COMPREHENSIVE INTEGRATION TEST
# ==============================================================================

def run_nlp_pipeline_demonstration():
    print("=" * 80)
    print("                   STARTING CUSTOM PYTHON NLP ENGINE RUNTIME                    ")
    print("=" * 80)

    # --- Dataset Blueprint Configuration ---
    raw_corpus = [
        "The deep learning architecture and neural networks are upgrading artificial intelligence rapidly.",
        "Natural Language Processing helps software applications understand human conversational language models.",
        "Generative Artificial Intelligence scales computer systems using smart transformer engineering modules.",
        "Stock market indices crashed yesterday following macroeconomic updates and high interest rates.",
        "Bonds, equity holdings, portfolio investment strategies, and financial assets yield high returns.",
        "Wall Street analysts project market volatility to scale over the upcoming fiscal quarters."
    ]
    
    # 0 for Tech/AI category, 1 for Finance/Markets category
    labels = [0, 0, 0, 1, 1, 1] 

    # --- Step 1: Preprocessing Pipeline ---
    print("\n[STEP 1] Initializing Core Text Preprocessor Engine...")
    processor = TextPreprocessor(extra_stopwords={"systems", "modules"})
    
    tokenized_corpus = []
    for document in raw_corpus:
        processed_tokens = processor.pipeline(document)
        tokenized_corpus.append(processed_tokens)
        
    print(f"-> Successfully processed {len(tokenized_corpus)} corpus entries.")
    print(f"-> Sample Tokenized Record Output: {tokenized_corpus[0]}")

    # --- Step 2: TF-IDF Extraction Execution ---
    print("\n[STEP 2] Building Term Frequency-Inverse Document Frequency Features...")
    tfidf_vectorizer = TFIDFVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_corpus)
    
    print(f"-> Generated Dense Matrix Structural Shapes: {tfidf_matrix.shape}")
    print(f"-> Vocabulary Size: {len(tfidf_vectorizer.vocabulary)} unique attributes learned.")
    print(f"-> Sample non-zero weights from index 0: {tfidf_matrix[0][tfidf_matrix[0] > 0][:4]}")

    # --- Step 3: N-Gram Modeling and Sequence Paths Execution ---
    print("\n[STEP 3] Training Bigram Generative Language Model Architecture...")
    language_model = NGramLanguageModel(n=2)
    language_model.train(tokenized_corpus)
    
    seed_phrase = "deep learning"
    generated_output = language_model.generate_text(seed_phrase, max_words=8)
    print(f"-> Input Seed Phrase: '{seed_phrase}'")
    print(f"-> Synthesized Target Output sequence path: '{generated_output}'")

    # --- Step 4: Semantic Topology Embedding Layouts ---
    print("\n[STEP 4] Constructing Geometric Spatial Word Embeddings Engine...")
    embedding_space = WordEmbeddingSpace()
    all_known_words = list(tfidf_vectorizer.vocabulary.keys())
    
    # Injection of specific relational tokens for analogy validations
    analogy_tokens = ["king", "queen", "man", "woman"]
    all_known_words.extend(analogy_tokens)
    
    embedding_space.generate_mock_embeddings(all_known_words)
    
    # Overriding vector spaces deliberately to construct geometric analogies
    # Vector(king) - Vector(man) + Vector(woman) should closely approximate Vector(queen)
    embedding_space.embeddings["man"] = np.array([0.5, 0.1] + [0.0]*98)
    embedding_space.embeddings["king"] = np.array([0.5, 0.9] + [0.0]*98)
    embedding_space.embeddings["woman"] = np.array([0.1, 0.1] + [0.0]*98)
    embedding_space.embeddings["queen"] = np.array([0.1, 0.85] + [0.0]*98) # Geometrically proximate destination
    
    # Normalize our custom overridden testing vectors
    for k in analogy_tokens:
        embedding_space.embeddings[k] /= np.linalg.norm(embedding_space.embeddings[k])

    target_similarity_query = "learning"
    neighbors = embedding_space.find_most_similar(target_similarity_query, top_n=2)
    print(f"-> Extracted Neighbors for '{target_similarity_query}': {neighbors}")
    
    analogy_solution = embedding_space.solve_analogy(a="man", b="king", c="woman", top_n=1)
    print(f"-> Solving Classic Semantic Equivalence Riddle: [ man : king :: woman : ? ]")
    print(f"-> Solved Analogy Output Target: {analogy_solution}")

    # --- Step 5: Supervised Multinomial Naive Bayes Execution ---
    print("\n[STEP 5] Allocating Supervised Multinomial Naive Bayes Classifier...")
    classifier = NaiveBayesClassifier(alpha=1.0)
    classifier.train(tokenized_corpus, labels)
    
    # Evaluate accuracy across internal training sets
    predictions = classifier.predict(tokenized_corpus)
    correct_matches = sum(1 for p, y in zip(predictions, labels) if p == y)
    training_accuracy = (correct_matches / len(labels)) * 100
    print(f"-> Finalized Training Accuracy Matrix calculation: {training_accuracy:.2f}%")
    
    # Evaluate generalization performance via an unseen test phrase string
    unseen_test_phrase = "Neural networks evaluate modern market portfolios and conversational financial models."
    processed_test_tokens = processor.pipeline(unseen_test_phrase)
    predicted_class = classifier.predict_single(processed_test_tokens)
    
    class_map = {0: "Technology & AI Science", 1:
