import re
import unicodedata
from dataclasses import dataclass

from django.conf import settings

DEFAULT_RAPTOR_MULTI_INTENT_ENABLED = True
DEFAULT_RAPTOR_MAX_SUBINTENTS = 4

INTERROGATIVE_RESTART_SPLIT_PATTERN = re.compile(
    r"\s*(?:[,;]\s*|\bet\s+)"
    r"(?=(?:quel|quelle|quels|quelles|combien|quand|ou)\b)",
    re.IGNORECASE,
)
IMPERATIVE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:donnez|donne|indiquez|indique|precisez|precise|listez|liste|resumez|resume)"
    r"(?:\s*:|\s+)",
    re.IGNORECASE,
)
LIST_SEPARATOR_PATTERN = re.compile(r"\s*[,;]\s*")
FINAL_ET_PATTERN = re.compile(r"\s+\bet\s+", re.IGNORECASE)
TRAILING_PUNCTUATION_PATTERN = re.compile(r"[\s.?!;:,]+$")

PROTECTED_COORDINATION_PATTERNS = {
    "date et heure limite",
    "date et heure limites",
    "date et l heure limite",
    "date et l heure limites",
    "nom et adresse",
    "cout d exploitation et maintenance",
    "couts d exploitation et maintenance",
    "cout d exploitation et de maintenance",
    "couts d exploitation et de maintenance",
}


@dataclass(frozen=True)
class RaptorIntentPlan:
    multi_intent: bool
    subintents: tuple[str, ...]
    strategy: str


def _get_bool_setting(name, default):
    value = getattr(settings, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _get_positive_int_setting(name, default):
    try:
        value = int(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default
    return max(1, value)


def get_raptor_multi_intent_enabled():
    return _get_bool_setting(
        "RAPTOR_MULTI_INTENT_ENABLED",
        DEFAULT_RAPTOR_MULTI_INTENT_ENABLED,
    )


def get_raptor_max_subintents():
    return _get_positive_int_setting(
        "RAPTOR_MAX_SUBINTENTS",
        DEFAULT_RAPTOR_MAX_SUBINTENTS,
    )


def _clean_question_text(text):
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\x00", "")
    cleaned = "".join(
        character
        for character in cleaned
        if character in {"\n", "\t"} or ord(character) >= 32
    )
    return " ".join(cleaned.split()).strip()


def _fold_text(text):
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    without_marks = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    folded = without_marks.lower()
    return " ".join(re.sub(r"[^0-9a-z]+", " ", folded).split())


def _capitalize_first(text):
    if not text:
        return text
    return text[0].upper() + text[1:]


def _normalize_subintent(text):
    normalized = TRAILING_PUNCTUATION_PATTERN.sub("", _clean_question_text(text))
    if not normalized:
        return ""
    normalized = _capitalize_first(normalized)
    return f"{normalized} ?"


def _question_from_list_fragment(fragment):
    normalized = TRAILING_PUNCTUATION_PATTERN.sub("", _clean_question_text(fragment))
    if not normalized:
        return ""
    folded = _fold_text(normalized)
    if folded.startswith(("les ", "des ")):
        return _normalize_subintent(f"Quelles sont {normalized}")
    if folded.startswith(("le ", "un ", "montant ", "nom ", "delai ")):
        return _normalize_subintent(f"Quel est {normalized}")
    return _normalize_subintent(f"Quelle est {normalized}")


def _dedupe_subintents(subintents, limit):
    unique = []
    seen = set()
    for subintent in subintents:
        normalized = _normalize_subintent(subintent)
        if not normalized:
            continue
        key = _fold_text(normalized)
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
        if len(unique) >= limit:
            break
    return tuple(unique)


def _contains_protected_coordination(text):
    folded = _fold_text(text)
    return any(pattern in folded for pattern in PROTECTED_COORDINATION_PATTERNS)


def _split_interrogative_restarts(question, limit):
    parts = [
        part
        for part in INTERROGATIVE_RESTART_SPLIT_PATTERN.split(question)
        if _clean_question_text(part)
    ]
    if len(parts) < 2:
        return ()
    return _dedupe_subintents(parts, limit)


def _split_imperative_list(question, limit):
    if not LIST_SEPARATOR_PATTERN.search(question):
        return ()
    match = IMPERATIVE_PREFIX_PATTERN.match(question)
    if not match:
        return ()

    body = question[match.end() :]
    fragments = []
    for part in LIST_SEPARATOR_PATTERN.split(body):
        part = _clean_question_text(part)
        if not part:
            continue
        if _contains_protected_coordination(part):
            fragments.append(part)
            continue
        fragments.extend(
            fragment
            for fragment in FINAL_ET_PATTERN.split(part)
            if _clean_question_text(fragment)
        )

    subintents = []
    seen = set()
    for fragment in fragments:
        subintent = _question_from_list_fragment(fragment)
        key = _fold_text(subintent)
        if not subintent or key in seen:
            continue
        seen.add(key)
        subintents.append(subintent)
        if len(subintents) >= limit:
            break
    return tuple(subintents) if len(subintents) >= 2 else ()


def detect_raptor_intents(question):
    normalized = _clean_question_text(question)
    if not normalized:
        return RaptorIntentPlan(False, (), "empty")

    max_subintents = get_raptor_max_subintents()
    if not get_raptor_multi_intent_enabled() or max_subintents < 2:
        return RaptorIntentPlan(False, (normalized,), "single_intent")

    interrogative_subintents = _split_interrogative_restarts(
        normalized,
        max_subintents,
    )
    if len(interrogative_subintents) >= 2:
        return RaptorIntentPlan(
            True,
            interrogative_subintents,
            "interrogative_restarts",
        )

    imperative_subintents = _split_imperative_list(normalized, max_subintents)
    if len(imperative_subintents) >= 2:
        return RaptorIntentPlan(True, imperative_subintents, "imperative_list")

    return RaptorIntentPlan(False, (normalized,), "single_intent")
