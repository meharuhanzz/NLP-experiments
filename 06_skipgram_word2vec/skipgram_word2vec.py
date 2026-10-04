"""Experiment 06: Skip-gram word2vec with negative sampling, from scratch.

Plain numpy, no gensim/PyTorch. Trains two embedding matrices (word
and context) on a toy corpus built from three distinct topics (animals,
food, tech), so a correct embedding space should visibly cluster words
by topic -- which is exactly what we check at the end via cosine
similarity, without needing any plotting library.
"""
import random
import numpy as np

random.seed(0)
np.random.seed(0)

CORPUS = [
    "the dog chased the cat across the yard",
    "the cat climbed the tree to escape the dog",
    "the puppy played with the kitten in the garden",
    "a wolf and a fox are wild relatives of the dog",
    "the horse and the cow grazed in the field",
    "the pizza had cheese tomato and fresh basil",
    "we ordered pasta with tomato sauce and cheese",
    "the chef cooked rice with vegetables and spices",
    "fresh bread with butter and jam tastes great",
    "the recipe needed flour sugar eggs and butter",
    "the laptop has a fast processor and good battery",
    "the new smartphone features a better camera and chip",
    "the software update improved the operating system speed",
    "engineers designed a faster chip for the new laptop",
    "the programmer wrote code to fix the software bug",
] * 6  # repeat so co-occurrence counts are high enough to learn from


def build_vocab(corpus):
    words = sorted(set(w for sent in corpus for w in sent.split()))
    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def build_training_pairs(corpus, word2idx, window=2):
    pairs = []
    for sent in corpus:
        tokens = [word2idx[w] for w in sent.split()]
        for i, center in enumerate(tokens):
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if j != i:
                    pairs.append((center, tokens[j]))
    return pairs


def negative_sampling_table(word2idx, corpus, power=0.75):
    counts = np.zeros(len(word2idx))
    for sent in corpus:
        for w in sent.split():
            counts[word2idx[w]] += 1
    probs = counts ** power
    probs /= probs.sum()
    return probs


class SkipGramNS:
    def __init__(self, vocab_size, dim=20, lr=0.05):
        self.W_in = np.random.randn(vocab_size, dim) * 0.1   # center-word embeddings
        self.W_out = np.random.randn(vocab_size, dim) * 0.1  # context-word embeddings
        self.lr = lr

    def train_pair(self, center, context, neg_samples):
        v_c = self.W_in[center]

        # positive pair: push sigmoid(v_c . v_context) towards 1
        u_pos = self.W_out[context]
        score_pos = sigmoid(np.dot(v_c, u_pos))
        grad_pos = (score_pos - 1.0) * self.lr
        grad_v = grad_pos * u_pos
        self.W_out[context] -= grad_pos * v_c

        # negative samples: push sigmoid(v_c . v_neg) towards 0
        for neg in neg_samples:
            u_neg = self.W_out[neg]
            score_neg = sigmoid(np.dot(v_c, u_neg))
            grad_neg = score_neg * self.lr
            grad_v += grad_neg * u_neg
            self.W_out[neg] -= grad_neg * v_c

        self.W_in[center] -= grad_v

    def embedding(self, idx):
        return self.W_in[idx]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def nearest_neighbors(model, word2idx, idx2word, word, k=5):
    target = model.embedding(word2idx[word])
    sims = []
    for w, idx in word2idx.items():
        if w == word:
            continue
        sims.append((w, cosine_sim(target, model.embedding(idx))))
    sims.sort(key=lambda x: -x[1])
    return sims[:k]


if __name__ == "__main__":
    word2idx, idx2word = build_vocab(CORPUS)
    pairs = build_training_pairs(CORPUS, word2idx, window=2)
    neg_probs = negative_sampling_table(word2idx, CORPUS)
    vocab_size = len(word2idx)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Training pairs: {len(pairs)}")

    model = SkipGramNS(vocab_size, dim=20, lr=0.05)
    epochs = 15
    for epoch in range(epochs):
        random.shuffle(pairs)
        for center, context in pairs:
            neg_samples = np.random.choice(vocab_size, size=5, p=neg_probs)
            model.train_pair(center, context, neg_samples)

    print("\n--- nearest neighbors after training ---")
    for query in ["dog", "cheese", "laptop"]:
        neighbors = nearest_neighbors(model, word2idx, idx2word, query, k=5)
        print(f"\n'{query}' is most similar to:")
        for w, sim in neighbors:
            print(f"   {w:12s} cosine={sim:.3f}")

    print("\n--- sanity check: cross-topic similarity should be lower ---")
    dog_cat = cosine_sim(model.embedding(word2idx["dog"]), model.embedding(word2idx["cat"]))
    dog_laptop = cosine_sim(model.embedding(word2idx["dog"]), model.embedding(word2idx["laptop"]))
    print(f"cosine(dog, cat)    = {dog_cat:.3f}   (same topic: animals)")
    print(f"cosine(dog, laptop) = {dog_laptop:.3f}   (different topic: tech)")
