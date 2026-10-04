"""Experiment 01: Tokenization strategies, compared.

Three tokenizers of increasing care, run on the same messy text
(contractions, punctuation, a URL, a hyphenated word, numbers):

  1. naive_split      -- str.split() on whitespace only
  2. punct_split       -- splits on whitespace AND peels off punctuation
  3. regex_tokenize    -- a single regex that treats contractions,
                          URLs, and numbers as atomic tokens

The point: naive whitespace splitting looks fine until text gets messy,
and "better" tokenization is really a series of judgment calls about
what counts as one token.
"""
import re

TEXT = (
    "Don't underestimate NLP preprocessing -- it's 80% of the work! "
    "Visit https://example.com/docs for details, or call us at 555-0199. "
    "We've processed 1,234 documents since 2024, and well-formed text "
    "still breaks naive tokenizers."
)


def naive_split(text):
    return text.split()


def punct_split(text):
    tokens = []
    for word in text.split():
        # peel leading/trailing punctuation off, keep the core token
        stripped = word.strip(".,!?;:\"'()")
        if stripped:
            tokens.append(stripped)
    return tokens


TOKEN_RE = re.compile(
    r"https?://\S+"          # URLs stay whole
    r"|\d[\d,]*\.?\d*"       # numbers (with commas/decimals) stay whole
    r"|\w+'\w+"              # contractions like "don't", "we've"
    r"|\w+(?:-\w+)*"         # hyphenated words like "well-formed"
    r"|[^\w\s]"              # anything else punctuation, one char at a time
)


def regex_tokenize(text):
    return TOKEN_RE.findall(text)


def show(name, tokens):
    print(f"\n--- {name} ({len(tokens)} tokens) ---")
    print(tokens)


if __name__ == "__main__":
    print("Input text:")
    print(TEXT)

    naive = naive_split(TEXT)
    punct = punct_split(TEXT)
    regex = regex_tokenize(TEXT)

    show("naive_split", naive)
    show("punct_split", punct)
    show("regex_tokenize", regex)

    print("\n--- where they disagree ---")
    glued = [t for t in naive if t not in punct and t.rstrip(".,!?;:\"'()") in punct]
    print(f"naive_split leaves trailing punctuation glued on, e.g.: {glued}")
    print("punct_split fixes that, but has no opinion on internal structure "
          "-- it happens to leave the URL and the hyphenated number alone "
          "only because strip() only touches the ends of a whitespace-delimited chunk.")

    print("\nregex_tokenize is more deliberate about URLs/contractions, but")
    print("its own corner-cutting shows up elsewhere:")
    print(f"  - '80%' got split into separate tokens: "
          f"{[t for t in regex if t in ('80', '%')]}")
    print(f"  - the number pattern greedily grabbed a trailing comma that "
          f"was just punctuation, not a thousands separator: "
          f"{[t for t in regex if t == '2024,']} (compare to the correctly-kept "
          f"thousands separator in {[t for t in regex if t == '1,234']})")
