# NLP Experiments

Eight small, self-contained NLP experiments, simple to medium
difficulty. Each one is a single runnable script with everything it
needs baked in — no external datasets, no network access, no large
model downloads. Where it's useful to see the actual math rather than
a library call, the technique is implemented from scratch (n-gram
smoothing, Naive Bayes, HMM Viterbi decoding, skip-gram negative
sampling, PageRank); where the standard tool is the point, it's used
directly (`TfidfVectorizer`, `LogisticRegression`).

Every script below was actually run to produce the output quoted —
nothing here is a guess at what the result "should" look like.

```bash
pip install -r requirements.txt
cd 01_tokenization && python tokenize_compare.py
```

## Contents

| # | Experiment | Level | Idea |
|---|---|---|---|
| 01 | [Tokenization](01_tokenization/) | Simple | Why whitespace-splitting breaks, and where a "smarter" regex tokenizer introduces its own new mistakes |
| 02 | [N-gram language model](02_ngram_language_model/) | Simple–medium | Bigram/trigram models with Laplace smoothing; perplexity; text generation; what an unseen n-gram does without smoothing |
| 03 | [TF-IDF classification](03_tfidf_text_classification/) | Simple | Headline topic classification with `TfidfVectorizer` + `LogisticRegression`, inspecting which words drive each class |
| 04 | [Naive Bayes sentiment](04_naive_bayes_sentiment/) | Medium | Multinomial Naive Bayes implemented from scratch, numerically cross-checked against scikit-learn on the same data |
| 05 | [HMM POS tagger](05_hmm_pos_tagger/) | Medium | Viterbi decoding vs. a greedy per-word baseline, on a case engineered to contain real tag ambiguity |
| 06 | [Skip-gram word2vec](06_skipgram_word2vec/) | Medium | Word embeddings with negative sampling, trained with nothing but numpy; checks that topic clusters actually form |
| 07 | [Rule-based NER](07_rule_based_ner/) | Simple–medium | Regex + gazetteer entity extraction, scored with real precision/recall, including a case built to expose its blind spot |
| 08 | [TextRank summarization](08_textrank_summarization/) | Medium | Extractive summarization: TF-IDF sentence similarity + PageRank via power iteration, implemented by hand |

## 01 — Tokenization

Three tokenizers (naive whitespace split, punctuation-stripping split,
a hand-written regex) run on the same deliberately messy sentence
(contractions, a URL, a hyphenated word, a thousands-separated
number). The regex tokenizer gets URLs and contractions right, but
introduces its own artifacts — it splits `"80%"` into `80` / `%`, and
greedily swallows a sentence-ending comma into `2024,` as if it were
part of the number. The lesson: there's no tokenizer that's simply
"correct," only one with a different, deliberate set of trade-offs.

## 02 — N-gram language model

```
Unseen bigram ('rug', 'chased') never appears in training.
  P(chased | rug) with Laplace smoothing:   0.04000
  P(chased | rug) with NO smoothing:         0.00000  <- infinite perplexity
```

Trigram perplexity (12.4) came out higher than bigram (9.1) on this
tiny corpus — expected, since there's far less data per trigram
context to estimate from, which is exactly the sparsity problem
smoothing exists to paper over.

## 03 — TF-IDF classification

100% test accuracy on a 30-headline tech-vs-sports dataset (an easy,
cleanly-separable toy problem by design). More interesting than the
accuracy number: inspecting the model's own coefficients to see which
words it actually keyed on —

```
most 'tech'-indicative words:   ['new', 'laptops', 'software', 'company', 'outage', 'battery', 'chip', 'faster']
most 'sports'-indicative words: ['ahead', 'season', 'player', 'team', 'match', 'opener', 'coach', 'star']
```

— then testing it on two brand-new sentences it never saw, both
classified correctly.

## 04 — Naive Bayes sentiment

A from-scratch multinomial Naive Bayes (manual word counts, Laplace
smoothing, log-space scoring) gets 4/4 on a held-out test set, and —
more importantly — produces **identical predictions** to scikit-learn's
`MultinomialNB` trained on the same data:

```
scikit-learn predictions: ['pos', 'neg', 'pos', 'neg']
from-scratch predictions: ['pos', 'neg', 'pos', 'neg']
Agreement between from-scratch and scikit-learn: 4/4
```

That agreement is the actual point of this experiment — it's not
enough for a from-scratch implementation to "look reasonable," it
should match a trusted reference exactly.

## 05 — HMM POS tagger

The training data was deliberately given one genuinely ambiguous word
— `"fast"`, tagged ADV four times and ADJ once. An emission-only greedy
tagger (no transition probabilities) always guesses the majority tag,
ADV, regardless of context:

```
sentence: a fast cat ran
  gold:    ['DET', 'ADJ', 'NOUN', 'VERB']
  viterbi: ['DET', 'ADJ', 'NOUN', 'VERB']  OK
  greedy:  ['DET', 'ADV', 'NOUN', 'VERB']  has errors

Viterbi: 20/20 = 100.0%
Greedy:  19/20 = 95.0%
```

Viterbi gets it right because `DET -> ADJ -> NOUN` is a transition
pattern seen repeatedly in training, while `DET -> ADV` essentially
never occurs — this is the whole reason to use an HMM over a per-word
lookup table.

## 06 — Skip-gram word2vec

A from-scratch skip-gram model with negative sampling (just numpy, two
embedding matrices, manual gradient updates), trained on a toy corpus
built from three topics: animals, food, tech.

```
cosine(dog, cat)    = 0.672   (same topic: animals)
cosine(dog, laptop) = 0.602   (different topic: tech)
```

Nearest neighbors after training also land in the right topic
(`laptop` → `smartphone`; `cheese` → `tomato`, `sauce`, `pizza`),
confirming the embedding space actually organized itself around
meaning rather than just memorizing training pairs.

## 07 — Rule-based NER

Regex handles DATE and MONEY well — those have a predictable surface
form, so pattern-matching genuinely works. PERSON and ORG rely on
gazetteers (fixed name/suffix lists), which is the classic failure
mode of rule-based NER. A deliberately adversarial sentence makes this
concrete:

```
text: Elena Petrova left DeepMind on June 1, 2023.
  predicted: [('June 1, 2023', 'DATE')]
  gold:      [('DeepMind', 'ORG'), ('Elena Petrova', 'PERSON'), ('June 1, 2023', 'DATE')]
  recall=0.33
```

Neither "Elena Petrova" nor "DeepMind" is in the hand-written
gazetteer, so the system silently misses both — it only ever
recognizes what it was explicitly told about.

## 08 — TextRank summarization

An 11-sentence article about renewable energy, summarized down to its
top 4 sentences (28% of the original length) by building a TF-IDF
sentence-similarity graph and running PageRank — implemented as plain
numpy power iteration, not a call into a graph library:

```
 - Renewable energy adoption has accelerated rapidly over the past decade.
 - Wind energy has also seen significant cost reductions during this period.
 - Grid operators are investing heavily in storage infrastructure.
 - Public support for renewable energy remains high in most surveyed countries.
```

The selected sentences read as a coherent summary on their own, which
is the real test of whether the ranking worked — a high PageRank score
that didn't correspond to actually-central sentences would show up
immediately as a summary that didn't hang together.
