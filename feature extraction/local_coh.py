import spacy
from typing import List, Set, Dict

nlp = spacy.load("en_core_web_sm")


def get_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def get_nouns(doc) -> Set[str]:
    return {token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"}}


def get_pronouns(doc) -> Set[str]:
    return {token.text.lower() for token in doc if token.pos_ == "PRON"}


def get_content_words(doc) -> Set[str]:
    return {
        token.text.lower()
        for token in doc
        if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
        and not token.is_stop
    }


def compute_overlap(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_coherence_features(text: str) -> Dict[str, float]:
    sentences = get_sentences(text)

    if len(sentences) < 2:
        return {
            "noun_overlap": None,
            "pronoun_overlap": None,
            "content_overlap": None,
        }

    noun_overlaps = []
    pronoun_overlaps = []
    content_overlaps = []

    for i in range(len(sentences) - 1):
        doc_a = nlp(sentences[i])
        doc_b = nlp(sentences[i + 1])

        nouns_a = get_nouns(doc_a)
        nouns_b = get_nouns(doc_b)

        pron_a = get_pronouns(doc_a)
        pron_b = get_pronouns(doc_b)

        content_a = get_content_words(doc_a)
        content_b = get_content_words(doc_b)

        noun_overlaps.append(compute_overlap(nouns_a, nouns_b))
        pronoun_overlaps.append(compute_overlap(pron_a, pron_b))
        content_overlaps.append(compute_overlap(content_a, content_b))

    return {
        "noun_overlap": sum(noun_overlaps) / len(noun_overlaps),
        "pronoun_overlap": sum(pronoun_overlaps) / len(pronoun_overlaps),
        "content_overlap": sum(content_overlaps) / len(content_overlaps),
    }