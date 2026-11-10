"""Independently reproducible text-only feature extraction.

These features intentionally do not reuse the benchmark's undocumented
engineered columns. The rules are simple, explicit, and runnable on arbitrary
English text. They are a candidate artifact for review, not the final model
schema.
"""

from __future__ import annotations

import math
import re
import unicodedata

import pandas as pd


WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
VOWELS = set("aeiouy")


def normalize_text(text: str) -> str:
    """Collapse Unicode whitespace runs and trim the text."""
    return WHITESPACE_PATTERN.sub(" ", str(text)).strip()


def word_tokens(text: str) -> list[str]:
    """Return maximal Unicode letter/number tokens, excluding underscores."""
    return WORD_PATTERN.findall(text)


def sentence_word_lengths(text: str) -> list[int]:
    """Count word tokens in nonempty spans split by terminal punctuation."""
    lengths = []
    for sentence in SENTENCE_SPLIT_PATTERN.split(text):
        count = len(word_tokens(sentence))
        if count:
            lengths.append(count)
    return lengths


def syllable_count(word: str) -> int:
    """Estimate English syllables using a documented vowel-group heuristic.

    The count is intentionally simple and deterministic: count vowel groups,
    subtract a final silent 'e' when there is more than one syllable, and clamp
    the result to one for nonempty words. This is not claimed to reproduce the
    benchmark's undocumented syllable implementation.
    """
    letters = "".join(character for character in word.lower() if character.isalpha())
    if not letters:
        return 0

    groups = 0
    previous_vowel = False
    for character in letters:
        is_vowel = character in VOWELS
        if is_vowel and not previous_vowel:
            groups += 1
        previous_vowel = is_vowel

    if letters.endswith("e") and groups > 1 and not letters.endswith(("le", "ye")):
        groups -= 1
    return max(1, groups)


def burstiness_from_lengths(lengths: list[int]) -> float:
    """Calculate population-SD burstiness for a sentence-length sequence."""
    if not lengths:
        return math.nan
    mean_length = sum(lengths) / len(lengths)
    standard_deviation = math.sqrt(
        sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
    )
    denominator = standard_deviation + mean_length
    return (standard_deviation - mean_length) / denominator if denominator else 0.0


def extract_text_features(text: str) -> dict[str, float | int]:
    """Extract the candidate feature vector from one arbitrary text string."""
    normalized = normalize_text(text)
    tokens = word_tokens(normalized)
    lowercase_tokens = [token.casefold() for token in tokens]
    word_count = len(tokens)
    unique_word_count = len(set(lowercase_tokens))
    sentence_lengths = sentence_word_lengths(normalized)
    sentence_count = len(sentence_lengths)
    character_count = len(normalized)
    punctuation_count = sum(
        unicodedata.category(character).startswith("P") for character in normalized
    )
    special_character_count = sum(
        not (
            character.isalpha()
            or character.isnumeric()
            or character.isspace()
            or unicodedata.category(character).startswith("P")
        )
        for character in normalized
    )
    capital_letter_count = sum(character.isalpha() and character.isupper() for character in normalized)

    if word_count == 0:
        return {
            "Text_Character_Count": character_count,
            "Text_Word_Count": 0,
            "Text_Sentence_Count": sentence_count,
            "Text_Unique_Word_Count": 0,
            "Text_Type_Token_Ratio": math.nan,
            "Text_Guiraud_Root_TTR": math.nan,
            "Text_Average_Word_Length": math.nan,
            "Text_Average_Sentence_Length_Words": math.nan,
            "Text_Punctuation_Count": punctuation_count,
            "Text_Punctuation_Density": math.nan,
            "Text_Special_Character_Count": special_character_count,
            "Text_Capital_Letter_Count": capital_letter_count,
            "Text_Average_Syllables_Per_Word": math.nan,
            "Text_Flesch_Kincaid_Grade_Level": math.nan,
            "Text_Burstiness": math.nan,
        }

    average_sentence_length = word_count / sentence_count if sentence_count else math.nan
    average_syllables = sum(syllable_count(token) for token in tokens) / word_count
    burstiness = math.nan
    if sentence_lengths:
        mean_length = sum(sentence_lengths) / len(sentence_lengths)
        standard_deviation = math.sqrt(
            sum((length - mean_length) ** 2 for length in sentence_lengths)
            / len(sentence_lengths)
        )
        denominator = standard_deviation + mean_length
        burstiness = (standard_deviation - mean_length) / denominator if denominator else 0.0

    return {
        "Text_Character_Count": character_count,
        "Text_Word_Count": word_count,
        "Text_Sentence_Count": sentence_count,
        "Text_Unique_Word_Count": unique_word_count,
        "Text_Type_Token_Ratio": unique_word_count / word_count,
        "Text_Guiraud_Root_TTR": unique_word_count / math.sqrt(word_count),
        "Text_Average_Word_Length": sum(map(len, tokens)) / word_count,
        "Text_Average_Sentence_Length_Words": average_sentence_length,
        "Text_Punctuation_Count": punctuation_count,
        "Text_Punctuation_Density": punctuation_count / max(1, character_count),
        "Text_Special_Character_Count": special_character_count,
        "Text_Capital_Letter_Count": capital_letter_count,
        "Text_Average_Syllables_Per_Word": average_syllables,
        "Text_Flesch_Kincaid_Grade_Level": (
            0.39 * average_sentence_length + 11.8 * average_syllables - 15.59
            if sentence_count
            else math.nan
        ),
        "Text_Burstiness": burstiness,
    }


def extract_dataframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one independently derived feature row for each input row.

    Pandas handles the simple string counts in bulk. Only the explicitly
    algorithmic syllable and sentence-length calculations remain row-wise.
    """
    normalized = df["Raw_Text"].map(normalize_text)
    tokens = normalized.str.findall(WORD_PATTERN)
    word_count = tokens.str.len()
    unique_word_count = tokens.map(lambda values: len({value.casefold() for value in values}))
    sentence_lengths = normalized.map(sentence_word_lengths)
    sentence_count = sentence_lengths.str.len()
    character_count = normalized.str.len()
    punctuation_count = normalized.map(
        lambda value: sum(unicodedata.category(character).startswith("P") for character in value)
    )
    special_character_count = normalized.map(
        lambda value: sum(
            not (
                character.isalpha()
                or character.isnumeric()
                or character.isspace()
                or unicodedata.category(character).startswith("P")
            )
            for character in value
        )
    )
    capital_letter_count = normalized.map(
        lambda value: sum(character.isalpha() and character.isupper() for character in value)
    )
    syllable_totals = tokens.map(lambda values: sum(syllable_count(value) for value in values))

    result = pd.DataFrame(index=df.index)
    result["Text_Character_Count"] = character_count
    result["Text_Word_Count"] = word_count
    result["Text_Sentence_Count"] = sentence_count
    result["Text_Unique_Word_Count"] = unique_word_count
    result["Text_Type_Token_Ratio"] = unique_word_count.div(word_count.replace(0, pd.NA))
    result["Text_Guiraud_Root_TTR"] = unique_word_count.div(word_count.replace(0, pd.NA).pow(0.5))
    result["Text_Average_Word_Length"] = tokens.map(
        lambda values: sum(map(len, values)) / len(values) if values else math.nan
    )
    result["Text_Average_Sentence_Length_Words"] = word_count.div(sentence_count.replace(0, pd.NA))
    result["Text_Punctuation_Count"] = punctuation_count
    result["Text_Punctuation_Density"] = punctuation_count.div(character_count.clip(lower=1))
    result["Text_Special_Character_Count"] = special_character_count
    result["Text_Capital_Letter_Count"] = capital_letter_count
    result["Text_Average_Syllables_Per_Word"] = syllable_totals.div(word_count.replace(0, pd.NA))
    result["Text_Flesch_Kincaid_Grade_Level"] = (
        0.39 * result["Text_Average_Sentence_Length_Words"]
        + 11.8 * result["Text_Average_Syllables_Per_Word"]
        - 15.59
    )
    result["Text_Burstiness"] = sentence_lengths.map(burstiness_from_lengths)
    return result
