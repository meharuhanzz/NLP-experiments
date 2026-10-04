"""Experiment 05: Hidden Markov Model POS tagger, with Viterbi decoding.

A small hand-tagged training set teaches transition probabilities
P(tag_i | tag_{i-1}) and emission probabilities P(word | tag). Viterbi
then finds the single most likely tag sequence for a new sentence --
not just the locally-best tag per word, which is the whole point of
using an HMM instead of a per-word lookup table.

Tagset kept deliberately tiny: DET, NOUN, VERB, ADJ, ADV, PRON, ADP
(adposition/preposition), CONJ, PUNCT.
"""
import math
from collections import defaultdict

TRAIN = [
    [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB"), ("on", "ADP"), ("the", "DET"), ("mat", "NOUN")],
    [("a", "DET"), ("dog", "NOUN"), ("ran", "VERB"), ("fast", "ADV")],
    [("the", "DET"), ("big", "ADJ"), ("dog", "NOUN"), ("chased", "VERB"), ("the", "DET"), ("cat", "NOUN")],
    [("she", "PRON"), ("quickly", "ADV"), ("saw", "VERB"), ("the", "DET"), ("cat", "NOUN")],
    [("the", "DET"), ("cat", "NOUN"), ("and", "CONJ"), ("the", "DET"), ("dog", "NOUN"), ("ran", "VERB")],
    [("he", "PRON"), ("sat", "VERB"), ("on", "ADP"), ("a", "DET"), ("big", "ADJ"), ("mat", "NOUN")],
    [("the", "DET"), ("old", "ADJ"), ("dog", "NOUN"), ("barked", "VERB"), ("loudly", "ADV")],
    [("she", "PRON"), ("saw", "VERB"), ("a", "DET"), ("small", "ADJ"), ("cat", "NOUN")],
    # "fast" is genuinely ambiguous: mostly seen as ADV here, but also ADJ once.
    [("a", "DET"), ("fast", "ADJ"), ("dog", "NOUN"), ("ran", "VERB")],
    [("the", "DET"), ("dog", "NOUN"), ("ran", "VERB"), ("fast", "ADV")],
    [("she", "PRON"), ("ran", "VERB"), ("fast", "ADV")],
    [("he", "PRON"), ("ran", "VERB"), ("fast", "ADV")],
]

TEST = [
    [("the", "DET"), ("small", "ADJ"), ("dog", "NOUN"), ("ran", "VERB"), ("quickly", "ADV")],
    [("he", "PRON"), ("saw", "VERB"), ("the", "DET"), ("old", "ADJ"), ("cat", "NOUN")],
    [("a", "DET"), ("cat", "NOUN"), ("sat", "VERB"), ("on", "ADP"), ("the", "DET"), ("dog", "NOUN")],
    # "fast" is ADV 4x and ADJ 1x in training -- an emission-only tagger will
    # always guess ADV for it. Here it should be ADJ: DET -> ADJ -> NOUN is a
    # transition pattern seen often in training, DET -> ADV essentially never.
    [("a", "DET"), ("fast", "ADJ"), ("cat", "NOUN"), ("ran", "VERB")],
]


class HMMTagger:
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing
        self.tag_counts = defaultdict(int)
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.emission_counts = defaultdict(lambda: defaultdict(int))
        self.tags = set()
        self.vocab = set()

    def fit(self, sentences):
        for sent in sentences:
            prev_tag = "<s>"
            for word, tag in sent:
                self.tags.add(tag)
                self.vocab.add(word)
                self.tag_counts[tag] += 1
                self.transition_counts[prev_tag][tag] += 1
                self.emission_counts[tag][word] += 1
                prev_tag = tag
            self.transition_counts[prev_tag]["</s>"] += 1
        self.tags = sorted(self.tags)

    def transition_prob(self, prev_tag, tag):
        counts = self.transition_counts[prev_tag]
        total = sum(counts.values())
        n_tags = len(self.tags) + 1  # +1 for </s>
        return (counts[tag] + self.smoothing) / (total + self.smoothing * n_tags)

    def emission_prob(self, tag, word):
        counts = self.emission_counts[tag]
        total = sum(counts.values())
        return (counts[word] + self.smoothing) / (total + self.smoothing * len(self.vocab))

    def viterbi(self, words):
        V = [{} for _ in words]
        backptr = [{} for _ in words]

        for tag in self.tags:
            V[0][tag] = math.log(self.transition_prob("<s>", tag)) + math.log(self.emission_prob(tag, words[0]))
            backptr[0][tag] = None

        for i in range(1, len(words)):
            for tag in self.tags:
                best_score, best_prev = max(
                    (V[i - 1][prev] + math.log(self.transition_prob(prev, tag)), prev)
                    for prev in self.tags
                )
                V[i][tag] = best_score + math.log(self.emission_prob(tag, words[i]))
                backptr[i][tag] = best_prev

        last_tag = max(V[-1], key=V[-1].get)
        path = [last_tag]
        for i in range(len(words) - 1, 0, -1):
            path.append(backptr[i][path[-1]])
        path.reverse()
        return path

    def greedy_baseline(self, words):
        """Per-word argmax over emission prob only, ignoring transitions --
        the naive approach Viterbi is meant to improve on."""
        return [max(self.tags, key=lambda t: self.emission_prob(t, w)) for w in words]


if __name__ == "__main__":
    hmm = HMMTagger(smoothing=0.1)
    hmm.fit(TRAIN)
    print(f"Tags learned: {hmm.tags}")
    print(f"Vocabulary size: {len(hmm.vocab)}")

    total_correct_viterbi = total_correct_greedy = total_words = 0
    for sent in TEST:
        words = [w for w, _ in sent]
        gold = [t for _, t in sent]
        viterbi_pred = hmm.viterbi(words)
        greedy_pred = hmm.greedy_baseline(words)

        print(f"\nsentence: {' '.join(words)}")
        print(f"  gold:    {gold}")
        print(f"  viterbi: {viterbi_pred}  {'OK' if viterbi_pred == gold else 'has errors'}")
        print(f"  greedy:  {greedy_pred}  {'OK' if greedy_pred == gold else 'has errors'}")

        total_correct_viterbi += sum(p == g for p, g in zip(viterbi_pred, gold))
        total_correct_greedy += sum(p == g for p, g in zip(greedy_pred, gold))
        total_words += len(words)

    print(f"\n--- per-word accuracy on held-out test sentences ---")
    print(f"Viterbi (uses transition probs): {total_correct_viterbi}/{total_words} "
          f"= {total_correct_viterbi/total_words:.1%}")
    print(f"Greedy (emission-only baseline): {total_correct_greedy}/{total_words} "
          f"= {total_correct_greedy/total_words:.1%}")
