"""Experiment 03: TF-IDF + Logistic Regression text classification.

A standard-tool exercise (unlike the from-scratch ones elsewhere in
this repo): use scikit-learn's TfidfVectorizer to turn short headlines
into vectors, then a Logistic Regression classifier to sort them into
"tech" vs "sports". Small hand-written dataset, fully offline.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

TECH_HEADLINES = [
    "Apple unveils new chip with faster AI processing",
    "Startup raises funding for cloud computing platform",
    "New smartphone features improved battery life",
    "Researchers build faster algorithm for data compression",
    "Tech company launches update for its operating system",
    "Engineers develop new chip architecture for laptops",
    "Software bug causes outage across cloud services",
    "Developers release open source tool for data analysis",
    "Company announces new processor for gaming laptops",
    "AI model achieves new benchmark in image recognition",
    "Browser update improves page load speed significantly",
    "New programming language gains popularity among developers",
    "Hackers exploit vulnerability in outdated software",
    "Company patents new battery technology for electric cars",
    "Server outage disrupts access to popular website",
]

SPORTS_HEADLINES = [
    "Team wins championship after dramatic overtime finish",
    "Star player scores winning goal in final minute",
    "Coach announces new lineup ahead of season opener",
    "Athlete breaks national record in long distance race",
    "Team signs new striker ahead of transfer deadline",
    "Underdog team shocks league leaders with upset win",
    "Player recovers from injury ahead of playoff match",
    "Fans celebrate as team clinches spot in finals",
    "Referee decision sparks controversy in tournament match",
    "Veteran player announces retirement after long career",
    "Team captain leads squad to victory in derby match",
    "New stadium opens ahead of upcoming football season",
    "Rookie player named most valuable player of tournament",
    "Coach praises team spirit after tough season opener",
    "Star athlete signs endorsement deal with sports brand",
]

texts = TECH_HEADLINES + SPORTS_HEADLINES
labels = ["tech"] * len(TECH_HEADLINES) + ["sports"] * len(SPORTS_HEADLINES)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)
preds = clf.predict(X_test_vec)

if __name__ == "__main__":
    print(f"Train set: {len(X_train)}, test set: {len(X_test)}")
    print(f"Vocabulary size after TF-IDF: {len(vectorizer.vocabulary_)}")
    print(f"\nAccuracy: {accuracy_score(y_test, preds):.3f}\n")
    print(classification_report(y_test, preds))

    print("--- predictions on the held-out test headlines ---")
    for text, true, pred in zip(X_test, y_test, preds):
        mark = "OK " if true == pred else "XX "
        print(f"{mark} true={true:7s} pred={pred:7s} | {text}")

    print("\n--- top TF-IDF-weighted words per class (by model coefficient) ---")
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]  # positive -> "tech" class (sklearn's class order)
    top_tech_idx = coefs.argsort()[-8:][::-1]
    top_sports_idx = coefs.argsort()[:8]
    print("most 'tech'-indicative words:  ", [feature_names[i] for i in top_tech_idx])
    print("most 'sports'-indicative words:", [feature_names[i] for i in top_sports_idx])

    print("\n--- trying it on a brand-new unseen headline ---")
    new_headlines = [
        "New laptop features longer battery and faster processor",
        "Team celebrates late goal to win local derby",
    ]
    new_vec = vectorizer.transform(new_headlines)
    new_preds = clf.predict(new_vec)
    for h, p in zip(new_headlines, new_preds):
        print(f"  [{p}] {h}")
