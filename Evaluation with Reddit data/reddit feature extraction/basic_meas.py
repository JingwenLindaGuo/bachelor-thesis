# -*- coding: utf-8 -*-
"""
Adapted from code by Dr Sarah E Morgan (21/02/2021), based on Morgan et al. (2021):
https://doi.org/10.1101/2021.01.04.20248717

Minor modifications in this version:
- added extract_basic_features()
- changed error handling to return None values instead of failing directly

Please cite Morgan et al. (2021) if you use this code or adapt it in your own work.
"""
import nltk
import string


# count the number of sentences in a text:
def get_no_sent(text):
    text = remove_text_inside_brackets(text)
    sentences = nltk.tokenize.sent_tokenize(text)
    return len(sentences)

# count the number of words in a text:
def get_no_words(text):
    text = remove_text_inside_brackets(text)
    tokens = nltk.word_tokenize(text)
    tokens = [t.lower() for t in tokens if t not in string.punctuation] # removes punctuation, so punctuation is not counted as a word
    return len(tokens)

# returns the total no. of words, total no. of sentences and mean sentence length for a text:
def meas_basic(text):
    totalnowords = get_no_words(text)
    totalnosent = get_no_sent(text)
    if totalnosent == 0:
        raise ValueError("sentence_count == 0")
    meansent = totalnowords / totalnosent

    return totalnowords, totalnosent, meansent

# removes text inside brackets, e.g. [?]
# taken from: https://stackoverflow.com/questions/14596884/remove-text-between-and
def remove_text_inside_brackets(text, brackets="()[]"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)

def extract_basic_features(text: str):
    if not text or not text.strip():
        return {
            "word_count": None,
            "sentence_count": None,
            "mean_sentence_length": None
        }

    try:
        total_words, total_sentences, mean_sent = meas_basic(text)

        return {
            "word_count": total_words,
            "sentence_count": total_sentences,
            "mean_sentence_length": mean_sent
        }

    except Exception as e:
        print("[basic feature error]", e)
        return {
            "word_count": None,
            "sentence_count": None,
            "mean_sentence_length": None
        }