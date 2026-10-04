"""Experiment 08: Extractive summarization with TextRank, from scratch.

Pipeline: split into sentences -> TF-IDF vectorize each sentence ->
build a sentence similarity graph (cosine similarity) -> rank
sentences with PageRank, implemented here as plain power iteration
over a numpy matrix, not a graph library call -> return the top-k
highest-ranked sentences in their original order as the summary.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ARTICLE = """
Renewable energy adoption has accelerated rapidly over the past decade.
Solar panel costs have dropped by more than eighty percent since 2010,
making solar power competitive with fossil fuels in many regions.
Wind energy has also seen significant cost reductions during this period.
Several countries now generate more than half of their electricity from
renewable sources. Battery storage technology has improved considerably,
helping to solve the problem of intermittent renewable generation.
Grid operators are investing heavily in storage infrastructure. Despite
this progress, fossil fuels still supply most of the world's energy.
Transitioning the remaining infrastructure will require significant
investment and political coordination. Experts estimate that full
decarbonization of the power grid could take another twenty years. Some
analysts argue that government policy, not just technology cost, is now
the main barrier to faster adoption. Public support for renewable energy
remains high in most surveyed countries.
""".strip()


def split_sentences(text):
    text = " ".join(text.split())
    # naive sentence splitter: split on '.' followed by a space and a
    # capital letter -- good enough for clean article text, would need
    # a real sentence boundary model for messier input (abbreviations etc).
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in raw if s.strip()]


def build_similarity_matrix(sentences):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(sentences).toarray()
    n = len(sentences)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            denom = np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
            sim[i, j] = np.dot(vectors[i], vectors[j]) / denom if denom > 0 else 0.0
    return sim


def pagerank(sim_matrix, damping=0.85, max_iter=100, tol=1e-6):
    n = sim_matrix.shape[0]
    # normalize rows so each sentence's outgoing similarity sums to 1
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = sim_matrix / row_sums

    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * transition.T.dot(scores)
        if np.abs(new_scores - scores).sum() < tol:
            scores = new_scores
            break
        scores = new_scores
    return scores


def summarize(text, top_k=4):
    sentences = split_sentences(text)
    sim_matrix = build_similarity_matrix(sentences)
    scores = pagerank(sim_matrix)

    ranked_idx = list(np.argsort(-scores))
    top_idx = sorted(ranked_idx[:top_k])  # restore original reading order
    summary = [sentences[i] for i in top_idx]
    return sentences, scores, summary


if __name__ == "__main__":
    sentences, scores, summary = summarize(ARTICLE, top_k=4)

    print(f"Article has {len(sentences)} sentences.\n")
    print("--- PageRank score per sentence ---")
    for i, (s, score) in enumerate(zip(sentences, scores)):
        print(f"[{i}] score={score:.4f}  {s}")

    print(f"\n--- extractive summary (top 4 of {len(sentences)} sentences, "
          f"original order) ---")
    for s in summary:
        print(" -", s)

    compression = len(" ".join(summary)) / len(ARTICLE)
    print(f"\nSummary is {compression:.0%} of the original character length.")
