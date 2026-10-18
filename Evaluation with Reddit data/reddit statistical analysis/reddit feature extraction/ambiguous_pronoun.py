from fastcoref import FCoref 

THIRD_PERSON_PRONOUNS = {
    "he", "him", "his",
    "she", "her", "hers",
    "they", "them", "their", "theirs",
    "it", "its"
}

ALL_PRONOUNS = THIRD_PERSON_PRONOUNS | {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves"
}


def is_pronoun(s: str) -> bool:
    return s.strip().lower() in ALL_PRONOUNS


def is_third_person_pronoun(s: str) -> bool:
    return s.strip().lower() in THIRD_PERSON_PRONOUNS


def compute_ambiguous_pronoun_features_from_pred(pred):
    """
    Compute ambiguous and unambiguous pronoun features directly
    from an existing prediction object, avoiding repeated model calls.
    """
    string_clusters = pred.get_clusters()
    span_clusters = pred.get_clusters(as_strings=False)

    ambiguous_mentions = []
    unambiguous_mentions = []

    for str_cluster, span_cluster in zip(string_clusters, span_clusters):
        mentions = sorted(zip(str_cluster, span_cluster), key=lambda x: x[1][0])

        for i, (mention_text, mention_span) in enumerate(mentions):
            if not is_third_person_pronoun(mention_text):
                continue

            earlier_non_pronoun = False

            for prev_text, prev_span in mentions[:i]:
                if not is_pronoun(prev_text):
                    earlier_non_pronoun = True
                    break

            mention_info = {
                "pronoun": mention_text,
                "span": mention_span,
                "cluster": str_cluster
            }

            if earlier_non_pronoun:
                unambiguous_mentions.append(mention_info)
            else:
                ambiguous_mentions.append(mention_info)

    return {
        "ambiguous_pronoun_count": len(ambiguous_mentions),
        "unambiguous_pronoun_count": len(unambiguous_mentions),
        "ambiguous_mentions": ambiguous_mentions,
        "unambiguous_mentions": unambiguous_mentions,
    }


coref_model = FCoref(device="cpu")


def extract_ambiguous_pronoun_count(text: str):
    if not text or not text.strip():
        return {"ambiguous_pronoun_count": None}

    try:
        preds = coref_model.predict(texts=[text])
        pred = preds[0]

        features = compute_ambiguous_pronoun_features_from_pred(pred)

        return {
            "ambiguous_pronoun_count": features["ambiguous_pronoun_count"]
        }

    except Exception as e:
        print("[coref error]", e)
        return {"ambiguous_pronoun_count": None}