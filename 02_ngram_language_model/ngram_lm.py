"""Experiment 02: N-gram language model with Laplace (add-1) smoothing.

Trains a bigram and a trigram model on a small synthetic corpus, then:
  - reports perplexity of each model on held-out text
  - generates random text by sampling from the learned distribution

This is a from-scratch implementation (just collections.Counter), not
a library call -- the point is to see exactly where smoothing enters
the probability formula and why unsmoothed models break the instant
they see an unseen n-gram.
"""
import random
from collections import Counter, defaultdict

random.seed(7)

TRAIN_TEXT = """
the cat sat on the mat . the dog sat on the rug .
the cat chased the dog . the dog chased the cat .
a cat and a dog can be friends . the mat was soft .
the rug was old . the cat liked the soft mat .
the dog liked the old rug . a happy cat sat near a happy dog .
""".strip()

TEST_TEXT = "the cat sat on the rug . the dog liked the mat ."


def tokenize(text):
    return text.split()


class NgramModel:
    def __init__(self, n, vocab_size, smoothing=1.0):
        self.n = n
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.context_counts = defaultdict(Counter)   # context -> {next_word: count}
        self.context_totals = defaultdict(int)        # context -> total count

    def train(self, tokens):
        padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1:i])
            word = padded[i]
            self.context_counts[context][word] += 1
            self.context_totals[context] += 1

    def prob(self, context, word):
        # Laplace smoothing: add `smoothing` to every count, and
        # `smoothing * vocab_size` to the denominator so probabilities
        # still sum to 1 over the vocabulary.
        count = self.context_counts[context][word]
        total = self.context_totals[context]
        return (count + self.smoothing) / (total + self.smoothing * self.vocab_size)

    def perplexity(self, tokens):
        padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        log_prob_sum = 0.0
        n_words = 0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1:i])
            word = padded[i]
            p = self.prob(context, word)
            log_prob_sum += -1.0 * _log2(p)
            n_words += 1
        avg_neg_log_prob = log_prob_sum / n_words
        return 2 ** avg_neg_log_prob

    def generate(self, max_len=15):
        context = tuple(["<s>"] * (self.n - 1))
        out = []
        for _ in range(max_len):
            candidates = self.context_counts.get(context)
            if not candidates:
                break
            words, weights = zip(*candidates.items())
            word = random.choices(words, weights=weights, k=1)[0]
            if word == "</s>":
                break
            out.append(word)
            context = tuple((list(context) + [word])[-(self.n - 1):]) if self.n > 1 else ()
        return " ".join(out)


def _log2(x):
    import math
    return math.log(x, 2)


if __name__ == "__main__":
    train_tokens = tokenize(TRAIN_TEXT)
    test_tokens = tokenize(TEST_TEXT)
    vocab = set(train_tokens) | {"<s>", "</s>"}
    print(f"Training tokens: {len(train_tokens)}, vocab size: {len(vocab)}")

    for n, name in [(2, "bigram"), (3, "trigram")]:
        model = NgramModel(n=n, vocab_size=len(vocab), smoothing=1.0)
        model.train(train_tokens)
        pp = model.perplexity(test_tokens)
        print(f"\n--- {name} model ---")
        print(f"Perplexity on held-out test text: {pp:.3f}")
        print("3 generated samples:")
        for _ in range(3):
            print("  ", model.generate())

    print("\n--- effect of smoothing ---")
    print("Unseen bigram ('rug', 'chased') never appears in training.")
    bigram_smoothed = NgramModel(n=2, vocab_size=len(vocab), smoothing=1.0)
    bigram_smoothed.train(train_tokens)
    bigram_unsmoothed = NgramModel(n=2, vocab_size=len(vocab), smoothing=0.0)
    bigram_unsmoothed.train(train_tokens)
    p_smoothed = bigram_smoothed.prob(("rug",), "chased")
    p_unsmoothed = bigram_unsmoothed.prob(("rug",), "chased")
    print(f"  P(chased | rug) with Laplace smoothing:   {p_smoothed:.5f}")
    print(f"  P(chased | rug) with NO smoothing:         {p_unsmoothed:.5f}  "
          f"<- zero probability means perplexity on any text containing this "
          f"pair would be infinite")
