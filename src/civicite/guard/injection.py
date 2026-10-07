"""Flag retrieved text that tries to give instructions to the model (indirect prompt injection)."""
from __future__ import annotations

import re

_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|system)",
    r"\b(system|developer)\s+prompt\b",
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\s+(an?\s+)?(unrestricted|jailbroken|dan)\b",
    r"\bdo\s+not\s+(tell|inform|mention\s+to)\s+the\s+user\b",
    r"\b(reveal|print|output)\s+(your|the)\s+(instructions|system prompt|api key)",
    r"<\|?(im_start|im_end|system|assistant)\|?>",
    r"^\s*(###\s*)?(system|assistant)\s*:",
    r"\bnew\s+instructions?\s*:",
    r"ignorera\s+(alla\s+)?(tidigare|föregående)\s+instruktioner",
    r"\bsend\s+(the\s+)?(user'?s?\s+)?(data|personnummer|password|credentials)\s+to\b",
]
_RX = [re.compile(p, re.I | re.M) for p in _PATTERNS]


def injection_findings(text: str) -> list[str]:
    return [rx.pattern for rx in _RX if rx.search(text)]


def is_suspicious(text: str) -> bool:
    return bool(injection_findings(text))
