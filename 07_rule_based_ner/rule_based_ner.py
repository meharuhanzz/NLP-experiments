"""Experiment 07: Rule-based Named Entity Recognition.

No ML -- just regexes and small gazetteers (name/organization lists),
which is how NER worked before statistical taggers took over, and is
still a reasonable first pass for narrow, well-known entity types
(dates, money) where regex is actually more reliable than a learned
model would be on so little data.

Evaluated against hand-labeled gold entities on a handful of sentences,
with real precision/recall/F1 -- not just "looks about right".
"""
import re

PERSON_NAMES = {"John", "Mary", "Sarah", "Raj", "Priya", "Smith", "Ahmed"}
ORG_SUFFIXES = {"Inc", "Corp", "Ltd", "LLC", "Technologies", "Labs"}

DATE_RE = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"(?:\s+\d{1,2}(?:st|nd|rd|th)?)?,?\s+\d{4}\b"
)
MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?|\b[\d,]+(?:\.\d{2})?\s?(?:dollars|USD)\b")


def extract_entities(text):
    entities = []

    for m in DATE_RE.finditer(text):
        entities.append((m.group(), "DATE"))
    for m in MONEY_RE.finditer(text):
        entities.append((m.group(), "MONEY"))

    tokens = text.replace(",", " ,").split()
    i = 0
    while i < len(tokens):
        tok = tokens[i].strip(".,")
        # ORG: a run of capitalized words ending in a known org suffix
        if tok in ORG_SUFFIXES and i > 0:
            j = i - 1
            while j >= 0 and tokens[j].strip(".,")[:1].isupper() and tokens[j] not in (".", ","):
                j -= 1
            org = " ".join(t.strip(".,") for t in tokens[j + 1:i + 1])
            entities.append((org, "ORG"))
            i += 1
            continue
        # PERSON: known first/last name, possibly followed by another capitalized word
        if tok in PERSON_NAMES:
            if i + 1 < len(tokens) and tokens[i + 1].strip(".,")[:1].isupper() and tokens[i + 1] not in ORG_SUFFIXES:
                entities.append((f"{tok} {tokens[i+1].strip(',.')}", "PERSON"))
                i += 2
                continue
            entities.append((tok, "PERSON"))
        i += 1

    return entities


TEST_SENTENCES = [
    {
        "text": "John Smith joined Acme Technologies on March 5, 2024 for a $75,000 salary.",
        "gold": {("John Smith", "PERSON"), ("Acme Technologies", "ORG"),
                 ("March 5, 2024", "DATE"), ("$75,000", "MONEY")},
    },
    {
        "text": "Sarah met Raj at Beta Corp to discuss a 12000 dollars contract.",
        "gold": {("Sarah", "PERSON"), ("Raj", "PERSON"), ("Beta Corp", "ORG"),
                 ("12000 dollars", "MONEY")},
    },
    {
        "text": "Priya Ahmed signed with Nova Labs for January 2025.",
        "gold": {("Priya Ahmed", "PERSON"), ("Nova Labs", "ORG"), ("January 2025", "DATE")},
    },
    {
        # Deliberately adversarial: a name NOT in the gazetteer, and an org
        # with no recognizable suffix. This is the classic failure mode of
        # rule-based NER -- it only knows what it was explicitly told about.
        "text": "Elena Petrova left DeepMind on June 1, 2023.",
        "gold": {("Elena Petrova", "PERSON"), ("DeepMind", "ORG"), ("June 1, 2023", "DATE")},
    },
]


def evaluate(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return tp, fp, fn, precision, recall, f1


if __name__ == "__main__":
    total_tp = total_fp = total_fn = 0

    for case in TEST_SENTENCES:
        pred = set(extract_entities(case["text"]))
        gold = case["gold"]
        tp, fp, fn, p, r, f1 = evaluate(pred, gold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"\ntext: {case['text']}")
        print(f"  predicted: {sorted(pred)}")
        print(f"  gold:      {sorted(gold)}")
        print(f"  precision={p:.2f} recall={r:.2f} f1={f1:.2f}")
        missed = gold - pred
        extra = pred - gold
        if missed:
            print(f"  missed: {sorted(missed)}")
        if extra:
            print(f"  false positives: {sorted(extra)}")

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) else 0.0
    print(f"\n--- overall across all test sentences ---")
    print(f"precision={overall_p:.2f} recall={overall_r:.2f} f1={overall_f1:.2f}")
