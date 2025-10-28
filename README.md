[![MasterHead](https://firebasestorage.googleapis.com/v0/b/flexi-coding.appspot.com/o/dempgi7-520f8d5f-63d4-4453-8822-dbc149ae27f8.gif?alt=media&token=91c0c7b2-93c3-4029-b011-1a8703c5730d)]()
<h1 align="center">🤖 Natural Language Processing (NLP) Practicals 🧠</h1>

<p align="center"><i>“Turning words into meaning — where text becomes intelligence.”</i></p>

<img align="right" alt="Coding" width="340" src="https://i.pinimg.com/originals/e4/26/70/e426702edf874b181aced1e2fa5c6cde.gif">

- 🌱 Implemented using **Python 3** and **NLTK**
- 💬 Includes **Corpus Handling, Tokenization, Stemming, Lemmatization, Normalization & POS Tagging**
- 🧩 Ideal for **NLP Beginners and MSBTE Students**
- 📫 Maintained by: **Krish Mhatre**

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=25&duration=2600&pause=700&color=4CC9F0&center=true&vCenter=true&width=600&lines=Exploring+Text+Corpora.;Mastering+Tokenization.;Unleashing+NLP+Power+with+Python.;Let's+Decode+Language!">
</p>

---

# 🛰 EXPERIMENT 1 — Text Corpus (Brown or Penn Treebank Corpus)

### 🎯 AIM  
Implement a program to use text corpus —  
**i)** Brown Corpus **OR** **ii)** Penn Treebank Corpus.

### ⚙️ REQUIREMENTS  
- Python 3  
- NLTK (`pip install nltk`)

### 💻 PROGRAM
import nltk
from nltk.corpus import brown

# Download and use Brown Corpus
nltk.download('brown')

# Display categories
print("Categories in Brown Corpus:\n", brown.categories())

# Display sample words
print("\nSample words from 'news' category:\n", brown.words(categories='news')[:20])
