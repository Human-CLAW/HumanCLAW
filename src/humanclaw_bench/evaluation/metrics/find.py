"""Find success rule used by the final full-validation table."""

from __future__ import annotations

import re

ALIASES = {
    "couch": ("couch", "sofa", "loveseat", "sectional"),
    "bed": ("bed", "mattress"),
    "toilet": ("toilet", "commode"),
    "chair": ("chair", "armchair", "stool", "seat"),
    "tv": ("tv", "television", "screen", "monitor"),
    "potted_plant": ("potted plant", "plant", "flower pot", "flowerpot"),
}
_NEGATION = re.compile(
    r"\b(no|not|cannot|can't|don't|isn't|aren't|n't|without|unable|none)\b",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def claims_target_visible(visible_state: str, category: str) -> bool:
    """Implement SSDMC's final rule-B subjective acknowledgement.

    Only a sentence that mentions the target (or a category alias) is relevant.
    That sentence must contain no negation token.  Looking at the whole response
    would incorrectly reject statements such as "the bed is visible; no chair is
    nearby", while ignoring negation would count "the bed is not visible".
    """

    aliases = ALIASES.get(category, (category.replace("_", " "),))
    text = str(visible_state or "").strip()
    for sentence in _SENTENCE_BOUNDARY.split(text):
        lowered = sentence.lower()
        mentions_target = "target" in lowered or any(
            alias in lowered for alias in aliases
        )
        if mentions_target and not _NEGATION.search(sentence):
            return True
    return False


__all__ = ["claims_target_visible"]
