"""How an answer to a counting question is read: the first integer in the answer; if there is none, the first number word.
An answer with neither is wrong. VQAv2 questions are scored with the official VQAv2 evaluator instead."""
import re

NUMBER_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def first_int(text):
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def parse_count(text):
    n = first_int(text)
    if n is not None:
        return n
    for word, i in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text.lower()):
            return i
    return None


def is_correct(answer_text, true_count):
    return parse_count(answer_text.strip().lower()) == int(true_count)
