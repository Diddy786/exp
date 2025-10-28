# Natural Language Processing (NLP) Experiments

This repository contains 5 fundamental NLP experiments implemented using Python and NLTK library. Each experiment demonstrates core concepts in natural language processing with practical implementations.

## 📋 Requirements

- **Python 3.x**
- **NLTK Library**
  ```bash
  pip install nltk
  ```

## 🔬 Experiments Overview

### 🔹 Experiment 1: Text Corpus (Brown Corpus)

**AIM:** Implement a program to use text corpus - Brown Corpus for linguistic analysis.

**Code:**
```python
import nltk
from nltk.corpus import brown

# Download and use Brown Corpus
nltk.download('brown')

# Display categories
print("Categories in Brown Corpus:\n", brown.categories())

# Display sample words
print("\nSample words from 'news' category:\n", brown.words(categories='news')[:20])
```

**Output:**
```
Categories in Brown Corpus:
['news', 'editorial', 'fiction', 'romance', 'science_fiction', ...]

Sample words from 'news' category:
['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', ...]
```

**Key Questions & Answers:**

**Q1. Define Brown Corpus. What's the main feature of Brown Corpus?**
- The Brown Corpus is the first structured text corpus of American English
- It is categorized into genres like news, fiction, and science, helping in linguistic analysis

**Q2. What's the difference between Brown Corpus and Penn Treebank Corpus?**

| Feature | Brown Corpus | Penn Treebank Corpus |
|---------|--------------|----------------------|
| Type | General English text | Annotated for syntactic parsing |
| Focus | Word frequency & grammar | Sentence structure (parse trees) |
| Use | Text classification | POS tagging & parsing |

---

### 🔹 Experiment 2: Sentence and Word Segmentation

**AIM:** Write a program for Sentence Segmentation and Word Segmentation using NLTK.

**Code:**
```python
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download necessary resources
nltk.download('punkt')
nltk.download('punkt_tab')

text = "Hello! Good Morning. Today is my NLP practical and I am very tensed."

# Sentence Segmentation
sentences = sent_tokenize(text)
print("Sentence Segmentation:", sentences)

# Word Segmentation
words = word_tokenize(text)
print("Word Segmentation:", words)
```

**Output:**
```
Sentence Segmentation: ['Hello!', 'Good Morning.', 'Today is my NLP practical and I am very tensed.']
Word Segmentation: ['Hello', '!', 'Good', 'Morning', '.', 'Today', 'is', 'my', 'NLP', 'practical', 'and', 'I', 'am', 'very', 'tensed', '.']
```

**Key Questions & Answers:**

**Q1. Split the text into word and sentence segmentation.**
✅ Implemented in the above code.

**Q2. State the library used for sentence segmentation and how to install that library.**
- **Library used:** NLTK
- **Installation command:** `pip install nltk`

---

### 🔹 Experiment 3: Lemmatization and Stemming Techniques

**AIM:** Apply Lemmatization and Stemming Techniques using Porter Stemmer and WordNet Lemmatizer.

**Code:**
```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary resources
nltk.download('wordnet')
nltk.download('omw-1.4')

text_words = ["running", "flies", "better", "easily", "studies"]

# Initialize stemmer and lemmatizer
ps = PorterStemmer()
lm = WordNetLemmatizer()

print("Stemming:")
for w in text_words:
    print(w, "→", ps.stem(w))

print("\nLemmatization:")
for w in text_words:
    print(w, "→", lm.lemmatize(w))
```

**Output:**
```
Stemming:
running → run
flies → fli
better → better
easily → easili
studies → studi

Lemmatization:
running → running
flies → fly
better → good
easily → easily
studies → study
```

**Key Questions & Answers:**

**Q1. What is the main difference between Stemming and Lemmatization?**

| Feature | Stemming | Lemmatization |
|---------|----------|---------------|
| Output | Root form (may not be valid word) | Real dictionary word |
| Method | Removes prefixes/suffixes | Uses linguistic rules |
| Example | "Studies" → "studi" | "Studies" → "study" |

**Q2. Name one scenario where Porter Stemmer might be preferred over Lancaster Stemmer.**
- When we need less aggressive stemming and better readability, Porter Stemmer is preferred.

---

### 🔹 Experiment 4: Text Normalization and Tokenization

**AIM:** Write a program for text normalization using NLTK — tokenizing text and removing special characters.

**Code:**
```python
import nltk
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')

text = "Hello!!! Welcome to NLP Practical #4 @2025."

# Remove special characters
clean_text = re.sub(r'[^A-Za-z0-9\s]', '', text)

# Tokenize
tokens = word_tokenize(clean_text)

print("Original Text:", text)
print("Cleaned Text:", clean_text)
print("Tokens:", tokens)
```

**Output:**
```
Original Text: Hello!!! Welcome to NLP Practical #4 @2025.
Cleaned Text: Hello Welcome to NLP Practical 4 2025
Tokens: ['Hello', 'Welcome', 'to', 'NLP', 'Practical', '4', '2025']
```

**Key Questions & Answers:**

**Q1. What is the purpose of removing special characters during text normalization?**
- It makes the text cleaner and easier for NLP models to process accurately.

**Q2. What is a Bigram? Give an example.**
- A Bigram is a pair of consecutive words.
- **Example:** "I love NLP" → ('I', 'love'), ('love', 'NLP')

---

### 🔹 Experiment 5: POS Tagging

**AIM:** Write a program for POS (Part-of-Speech) Tagging on the given text.

**Code:**
```python
import nltk
from nltk import word_tokenize, pos_tag

# Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

text = "NLP helps computers understand human language."

# Tokenize and tag
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

print("Tokens:", tokens)
print("POS Tags:", pos_tags)
```

**Output:**
```
Tokens: ['NLP', 'helps', 'computers', 'understand', 'human', 'language', '.']
POS Tags: [('NLP', 'NNP'), ('helps', 'VBZ'), ('computers', 'NNS'), ('understand', 'VB'), ('human', 'JJ'), ('language', 'NN'), ('.', '.')]
```

**Key Questions & Answers:**

**Q1. What is the role of the word_tokenize() function in the POS tagging program?**
- It splits the text into individual words for analysis.

**Q2. Why do we use nltk.download('averaged_perceptron_tagger') before performing POS tagging?**
- It downloads the pretrained model used for assigning POS tags.

---

## 🎯 Conclusions

1. **Text Corpus:** Successfully accessed and explored the Brown Corpus using NLTK, understanding how large annotated text datasets support NLP model training.

2. **Segmentation:** Sentence and word segmentation divide text into meaningful parts, which is the first step in text preprocessing.

3. **Stemming vs Lemmatization:** Stemming reduces words to root forms; Lemmatization provides valid dictionary words. Lemmatization is more accurate.

4. **Text Normalization:** Text normalization improves text consistency and readability by removing noise like symbols and punctuation.

5. **POS Tagging:** POS tagging helps identify the grammatical structure of sentences, aiding in syntactic and semantic analysis.

## 🚀 Getting Started

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install nltk
   ```
3. Run any experiment script
4. The necessary NLTK data will be downloaded automatically when you run the scripts

## 📚 Additional Resources

- [NLTK Documentation](https://www.nltk.org/)
- [Brown Corpus Information](https://www.nltk.org/book/ch02.html)
- [POS Tagging Guide](https://www.nltk.org/book/ch05.html)

---

**Note:** Each experiment includes practical implementations with real outputs and comprehensive Q&A sections to reinforce learning concepts.
