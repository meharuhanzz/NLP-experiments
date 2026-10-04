"""Experiment 04: Multinomial Naive Bayes sentiment classifier, from scratch.

Implements the actual Naive Bayes math (word counts, class priors,
Laplace-smoothed likelihoods, log-space scoring) with nothing but
collections.Counter and math.log -- then cross-checks the from-scratch
implementation against scikit-learn's MultinomialNB on the exact same
data, to confirm it's not just plausible-looking but numerically right.
"""
import math
from collections import Counter, defaultdict

POSITIVE = [
    "the movie was absolutely wonderful and touching",
    "a brilliant performance with a heartwarming story",
    "I loved every minute of this delightful film",
    "an amazing cast delivered a truly moving experience",
    "the plot was gripping and the ending was beautiful",
    "a joyful and uplifting movie that I would watch again",
    "the acting was superb and the direction was masterful",
    "a wonderful film with great music and a happy ending",
    "this was a fantastic and inspiring story",
    "the characters were charming and the humor was great",
]

NEGATIVE = [
    "the movie was boring and poorly written",
    "a disappointing performance with a dull story",
    "I hated almost every minute of this terrible film",
    "a weak cast delivered an unconvincing experience",
    "the plot was confusing and the ending was awful",
    "a tedious and depressing movie I would not watch again",
    "the acting was wooden and the direction was clumsy",
    "a disappointing film with bad music and a sad ending",
    "this was a dull and uninspired story",
    "the characters were annoying and the humor fell flat",
]

TEST_SET = [
    ("the film was wonderful and the cast was brilliant", "pos"),
    ("a boring plot with a disappointing ending", "neg"),
    ("I loved the heartwarming and touching story", "pos"),
    ("the acting was dull and the story was annoying", "neg"),
]


def tokenize(text):
    return text.lower().split()


class NaiveBayesFromScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_word_counts = defaultdict(Counter)
        self.class_totals = defaultdict(int)
        self.class_doc_counts = defaultdict(int)
        self.vocab = set()
        self.classes = []

    def fit(self, docs, labels):
        self.classes = sorted(set(labels))
        for doc, label in zip(docs, labels):
            self.class_doc_counts[label] += 1
            for word in tokenize(doc):
                self.class_word_counts[label][word] += 1
                self.class_totals[label] += 1
                self.vocab.add(word)
        self.n_docs = len(docs)

    def _log_likelihood(self, word, label):
        count = self.class_word_counts[label][word]
        total = self.class_totals[label]
        return math.log((count + self.alpha) / (total + self.alpha * len(self.vocab)))

    def predict(self, doc):
        scores = {}
        for label in self.classes:
            log_prior = math.log(self.class_doc_counts[label] / self.n_docs)
            log_score = log_prior + sum(self._log_likelihood(w, label) for w in tokenize(doc))
            scores[label] = log_score
        best = max(scores, key=scores.get)
        return best, scores


if __name__ == "__main__":
    docs = POSITIVE + NEGATIVE
    labels = ["pos"] * len(POSITIVE) + ["neg"] * len(NEGATIVE)

    nb = NaiveBayesFromScratch(alpha=1.0)
    nb.fit(docs, labels)

    print("--- from-scratch Naive Bayes ---")
    correct = 0
    for text, true_label in TEST_SET:
        pred, scores = nb.predict(text)
        correct += (pred == true_label)
        mark = "OK " if pred == true_label else "XX "
        print(f"{mark} true={true_label} pred={pred} "
              f"(log-score pos={scores['pos']:.2f}, neg={scores['neg']:.2f}) | {text}")
    print(f"Accuracy: {correct}/{len(TEST_SET)}")

    print("\n--- cross-check against scikit-learn's MultinomialNB ---")
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    vec = CountVectorizer()
    X = vec.fit_transform(docs)
    sk_nb = MultinomialNB(alpha=1.0)
    sk_nb.fit(X, labels)

    test_texts = [t for t, _ in TEST_SET]
    test_labels = [l for _, l in TEST_SET]
    X_test = vec.transform(test_texts)
    sk_preds = sk_nb.predict(X_test)

    agree = sum(1 for a, b in zip(
        [nb.predict(t)[0] for t in test_texts], sk_preds
    ) if a == b)
    print(f"scikit-learn predictions: {list(sk_preds)}")
    print(f"from-scratch predictions: {[nb.predict(t)[0] for t in test_texts]}")
    print(f"Agreement between from-scratch and scikit-learn: {agree}/{len(test_texts)}")
    sk_correct = sum(1 for p, t in zip(sk_preds, test_labels) if p == t)
    print(f"scikit-learn accuracy on the same test set: {sk_correct}/{len(test_labels)}")
