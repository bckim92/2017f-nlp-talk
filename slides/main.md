name: inverse
class: center, middle, inverse
layout: true

---
class: titlepage, no-number

# NTLK: the natural language toolkit

## .author[Byeongchang Kim]

### .small[.white[Sep 7th, 2017] <br/> ]

### .x-small[https://bckim92.github.io/2017f-nlp-talk]

---
layout: false

## Natural Language Processing

.footnote[(Slide credit: [Cioroianu's NLTK tutorial](http://www.nyu.edu/projects/politicsdatalab/localdata/workshops/NLTK_Presentation.pdf))]

- NLP
  - Broad sense: any kind of computer manipulation of natural language
  - From word frequencies to "understanding" meaning

--

- Applications
  - Text processing
  - Information extraction
  - Document classification and sentiment analysis
  - Document similarity
  - Automatic summarizing
  - Discourse analysis


---

## What is NLTK?

.center.img-66[![](images/nltk-main.png)]
.center.small[S. Bird et al. [NLTK: the natural language toolkt][bird-2006], In *COLING-ACL*, 2006]

- Suite of open source Python libraries and programes for NLP
  - Python: open source programming language
- Developed for educational purposes by Steven Bird, Ewan Klein and Edward Loper
- Very good online documentation

[bird-2006]: http://www.aclweb.org/anthology/P04-3031

---

## Some Numbers

- 3+ classification algorithms
- 9+ Part-of-Speech tagging Algorithms
- Stemming algorithms for 15+ languages
- 5+ word tokenization algorithms
- Sentence tokenizers for 16+ languages
- 60+ included corpora

.footnote[(Slide credit: [Perkins's slide](https://www.slideshare.net/japerk/nltk-the-good-the-bad-and-the-awesome-8556908))]

---

## The Good

- Preprocessing
  - segmentation, tokenization, PoS tagging
- Word level processing
  - WordNet, lemmatization, stemming, n-gram
- Utilities
  - Tree, FreqDist, ConditionalFreqDist
  - Streaming CorpusReader objects
- Classification
  - Maximum Entropy, Naive Bayes, Decision Tree
  - Chunking, Named Entity Recognition
- Parsers Galore!
- Languages Galore!

.footnote[(Slide credit: [Bengfort's slide](https://www.slideshare.net/BenjaminBengfort/natural-language-processing-with-nltk?next_slideshow=2))]

---

## The Bad

- Syntactic parsing
  - No included grammer (not a black box)
- Feature/dependency parsing
  - No included feature grammer
- The sem package
  - Toy only (lambda-calculus & first order logic)
- Lots of extra stuff
  - Papers, chat programs, alignments, etc

.footnote[(Slide credit: [Bengfort's slide](https://www.slideshare.net/BenjaminBengfort/natural-language-processing-with-nltk?next_slideshow=2))]

---

name: centertext
class: center, middle, centertext

## Let's try NLTK!

---

## Installation

```python
$ pip install -U nltk
$ python
>>> import nltk
>>> nltk.download()
```


---

## Corpus - `nltk.corpus` module

- Corpus
  - Large collection of text
  - Raw or categorized
  - Concentrate on a topic or open domain
- Examples
  - Brown - first, largest corpus, categorized by genre
  - Webtext - reviews, forums, etc
  - Reuters - news corpus
  - Inaugural - US presidents' inaugural addresses
  - udhr - multilingual
- Available corpora can be seen at [http://www.nltk.org/nltk_data/](http://www.nltk.org/nltk_data/)

.footnote[(Slide credit: [Cioroianu's NLTK tutorial](http://www.nyu.edu/projects/politicsdatalab/localdata/workshops/NLTK_Presentation.pdf))]

---

## Corpus - `nltk.corpus` module

For example, to read a list of the words in the Brownn Corpus, use `nltk.corpus.brown.words()`

```python
>>> from nltk.corpus import brown
>>> print(", ".join(brown.words()))
The, Fulton, County, Grand, Jury, said, ...
```

---

## Tokenization - `nltk.tokenize` module

A sentence can be split into words using `word_tokenize()`

```python
>>> from nltk.tokenize import word_tokenize, sent_tokenize
>>> sentence = "All work and no play makes jack a dull boy, all work and no play"
*>>> tokens = word_tokenize(sentence)
>>> tokens
['All', 'work', 'and', 'no', 'play', 'makes', 'jack', 'a',
'dull', 'boy', ',', 'all', 'work', 'no', 'play']
```

--

Same principle can be applied to sentences via `sent_tokenize()`
```python
>>> sentence = "All work and no play makes jack a dull boy. All work and no play"
*>>> tokens = sent_tokenize(sentence)
>>> tokens
['All work and no play makes jack dull boy.', 'All work and no play']
```

---

## Stemming and lemmatization - `nltk.stem` module

<!--
.center.img-33[![](images/word-stem.png)]
-->

Words can be stemmed via `Stemmer()`

```python
>>> from nltk.stem import PorterStemmer
>>> words = ["game", "gaming", "gamed", "games"]
*>>> ps = PorterStemmer()
>>> [ps.stem(word) for word in words]
['game', 'game', 'game', 'game']
```

--

We can also lemmatize via `WordNetLemmatizer()`

```python
>>> from nltk.stem import WordNetLemmatizer
>>> words = ["game", "gaming", "gamed", "games"]
*>>> wnl = WordNetLemmatizer()
>>> [wnl.lemmatize(word) for word in words]
['game', 'gaming', 'gamed', u'game']
```

---

## Tagging - `nltk.tag` module

<!--
.center.img-50[![](images/pos-tagging.jpg)]
-->

A sentence can be tagged using `Tagger()`

```python
>>> nltk.corpus import brown
>>> from nltk.tag import UnigramTagger
*>>> tagger = UnigramTagger()
>>> sent = ['Mitchell', 'decried', 'the', 'high', 'rate', 'of', 'unnemployment']
*>>> tagger.tag(sent)
[('Mitchell', u'NP'), ('decried', None), ('the', u'AT'), ('high', u'JJ'), ('rate', u'NN'), ('of', u'IN'), ('unemployment', None)]
```

--

Or simply use NLTK's recommended tagger via `pos_tag()`

```python
>>> from nltk.tag import pos_tag
>>> from nltk.tokenize import word_tokenize
*>>> pos_tag(word_tokenize("John's big idea isn't all that bad."))
[('John', 'NNP'), ("'s", 'POS'), ('big', 'JJ'), ('idea', 'NN'), ('is', 'VBZ'),
("n't", 'RB'), ('all', 'PDT'), ('that', 'DT'), ('bad', 'JJ'), ('.', '.')]
```

---

## Parsing - `nltk.parser` module

Provide wrapper for CoreNLP parser `CoreNLPParser()`

```python
>>> parser = CoreNLPParser(url='http://localhost:9000')
>>> next(
...     parser.raw_parse('The quick brown fox jumps over the lazy dog.')
... ).pretty_print()
                     ROOT
                      |
                      S
       _______________|__________________________
      |                         VP               |
      |                _________|___             |
      |               |             PP           |
      |               |     ________|___         |
      NP              |    |            NP       |
  ____|__________     |    |     _______|____    |
 DT   JJ    JJ   NN  VBZ   IN   DT      JJ   NN  .
 |    |     |    |    |    |    |       |    |   |
The quick brown fox jumps over the     lazy dog  .
```

---

## Other libraries?

- [spaCy](https://spacy.io/)
- [unidecode](https://github.com/iki/unidecode)
- [PyEnchant](http://pythonhosted.org/pyenchant/)
- [gensim](https://radimrehurek.com/gensim/)
- [fastText](https://github.com/facebookresearch/fastText)

---

## spaCy: Industrial-Strength NLP in Python

.center.img-66[![](images/spacy-main.png)]

- Minimal and optimized!
  - One algorithm (the best one) for each purpose
- Lightning-fast (written in Cython)

---

## Detailed Speed Comparision

Per-document processing time of various spaCy functionalities against other NLP libraries

.center.img-77[![](images/spacy-benchmark.png)]

---

## Parse Accuracy

Google's [SyntaxNet](https://github.com/tensorflow/models/tree/master/syntaxnet) is the winner

.center.img-77[![](images/spacy-parse-accuracy.png)]

---

## Named Entity Comparison

[Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) is the winner

.center.img-77[![](images/spacy-ner-accuracy.png)]

---

## unidecode

ASCII transliterations of Unicode text

```python
>>> from unidecode import unnidecode
>>> unidecode(u'파이콘')
'paikon'
```

--

## pyEnchant

Spellchecking library for python

```python
>>> import enchant
>>> en_dict = enchant.Dict("en_US")
>>> en_dict.check("apple")
True
```

---

## gensim

Python library for topic modelling

```python
>>> from gensim.models.ldamulticore import LdaMulticore
>>> model = LdaMulticore(corpus, id2word=id2word, num_topics=num_topics, ...)
```

--

## fastText

Library for fast text representation and classification

```bash
$ ./fasttext skipgram -input data.txt -output model
$ ./fasttext print-word-vectors model.bin < queries.txt
```
---

name: last-page
class: center, middle, no-number
## Thank You!
#### [@bckim92][bckim92-gh]

.footnote[Slideshow created using [remark](http://github.com/gnab/remark).]

[bckim92-gh]: https://github.com/bckim92


