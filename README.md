#  Topic modelling using Latent Dirichlet Allocation (LDA ) in python

## What is Topic Modeling?
Topic modeling is a type of statistical modeling used to discover the abstract "topics" that
occur in a collection of documents. Latent Dirichlet Allocation (LDA) is one of the most
popular algorithms for topic modeling. It assumes:
1. Each document is a mixture of topics.
2. Each topic is a mixture of words.

## How LDA Works:
1. Input: A collection of documents.
2. Output: A set of topics (clusters of words), where each document is associated
with a proportion of topics.
3. LDA uses a probabilistic approach:
  o It determines which words are more likely to belong to a specific topic.
  o For example, the topic "Machine Learning" might contain words like
model, algorithm, data, neural.

## Example:

### Dataset:
Consider 5 short documents:
1. "Machine learning is transforming artificial intelligence."
2. "Natural language processing is fascinating."
3. "Artificial intelligence and machine learning are related."
4. "Data science involves statistics and problem-solving."
5. "Deep learning uses neural networks."

### LDA Process:
1. Preprocessing: Clean the data (e.g., remove stopwords, punctuation,
lowercase, etc.), then tokenize and lemmatize it.
Example:
o Document 1 becomes: ['machine', 'learning', 'transform', 'artificial',
'intelligence']
2. Bag of Words (BoW): Create a frequency matrix where each document is
represented by word counts.
SQL code :
Document 1: machine=1, learning=1, transform=1, artificial=1, intelligence=1
Document 2: natural=1, language=1, processing=1, fascinating=1
3. Train LDA: Identify topics and their associated words.
o Number of Topics (k): Assume we want 2 topics.
o Output: Each topic is a collection of words with probabilities

### LDA Output:
#### Topics:
1. Topic 0: Words like learning, machine, artificial, intelligence (related to AI/ML).
2. Topic 1: Words like natural, language, processing, neural (related to NLP/Deep
Learning).


#### Document-Topic Distribution:
• Document 1 → Topic 0 (90%), Topic 1 (10%)
• Document 2 → Topic 1 (85%), Topic 0 (15%)
• Document 5 → Topic 1 (95%), Topic 0 (5%)

### Interpretation:
1. Topics: Each topic is represented by the most probable words.
o Topic 0: "learning", "machine", "artificial" (Machine Learning)
o Topic 1: "natural", "language", "processing" (NLP)
2. Documents:
o Document 1 is mostly about Topic 0.
o Document 2 is primarily about Topic 1.

## Python Code for LDA Example
```python
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Example dataset
documents = [
 "Machine learning is transforming artificial intelligence.",
 "Natural language processing is fascinating.",
 "Artificial intelligence and machine learning are related.",
 "Data science involves statistics and problem solving.",
 "Deep learning uses neural networks."
]

# Preprocessing function
def preprocess_text(text):
  stop_words = set(stopwords.words('english'))
  lemmatizer = WordNetLemmatizer()
  tokens = word_tokenize(text.lower())
  return [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]

# Preprocess the documents
processed_docs = [preprocess_text(doc) for doc in documents]

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Train the LDA model
lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Print topics
for idx, topic in lda_model.print_topics(num_words=5):
  print(f"Topic {idx}: {topic}")
```

## Example Output
```
Topic 0: 0.102*"learning" + 0.088*"machine" + 0.072*"artificial" + 0.051*"data" +
0.047*"intelligence"
Topic 1: 0.118*"processing" + 0.102*"language" + 0.091*"deep" + 0.068*"neural" +
0.063*"natural"
```

✓ LDA assigns words to topics based on probabilities.
✓ Applications: Topic modeling is widely used in recommendation systems,
document classification, and summarization.
✓ Interpretation: You get meaningful topics (clusters of words) from large,
unstructured text data.
