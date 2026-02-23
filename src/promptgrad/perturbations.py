"""
Prompt perturbation strategies for sensitivity analysis.
"""

from __future__ import annotations

import random
import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PerturbationResult:
    """A single perturbed variant of a prompt."""

    original: str
    perturbed: str
    strategy: str
    token_index: Optional[int] = None
    metadata: dict = field(default_factory=dict)


class Perturbation(ABC):
    """Abstract base class for perturbation strategies."""

    @abstractmethod
    def apply(self, prompt: str, n: int = 5, seed: Optional[int] = None) -> List[PerturbationResult]:
        """Generate `n` perturbed variants of `prompt`."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class SynonymSubstitution(Perturbation):
    """Replace words with simple lexical near-synonyms / paraphrases."""

    # Lightweight synonym map — no external dependencies required.
    _SYNONYMS: dict[str, list[str]] = {
        "provide": ["give", "supply", "return", "output"],
        "generate": ["create", "produce", "write", "build"],
        "summarize": ["condense", "recap", "overview", "describe"],
        "analyze": ["examine", "assess", "evaluate", "review"],
        "explain": ["describe", "clarify", "detail", "elaborate on"],
        "list": ["enumerate", "outline", "show", "display"],
        "write": ["compose", "draft", "produce", "create"],
        "find": ["locate", "identify", "determine", "discover"],
        "compare": ["contrast", "evaluate", "assess", "differentiate"],
        "help": ["assist", "support", "aid", "guide"],
        "make": ["create", "build", "construct", "produce"],
        "use": ["utilize", "employ", "apply", "leverage"],
        "show": ["display", "present", "demonstrate", "illustrate"],
        "tell": ["inform", "explain", "describe", "convey"],
        "get": ["obtain", "retrieve", "fetch", "acquire"],
        "good": ["effective", "high-quality", "solid", "strong"],
        "best": ["optimal", "top", "ideal", "most effective"],
        "important": ["critical", "key", "essential", "significant"],
        "simple": ["straightforward", "basic", "easy", "clear"],
        "detailed": ["thorough", "comprehensive", "in-depth", "extensive"],
    }

    @property
    def name(self) -> str:
        return "synonym_substitution"

    def apply(self, prompt: str, n: int = 5, seed: Optional[int] = None) -> List[PerturbationResult]:
        rng = random.Random(seed)
        words = prompt.split()
        replaceable = [
            (i, w, self._SYNONYMS[w.lower().strip(string.punctuation)])
            for i, w in enumerate(words)
            if w.lower().strip(string.punctuation) in self._SYNONYMS
        ]

        results = []
        for _ in range(n):
            if not replaceable:
                break
            idx, word, synonyms = rng.choice(replaceable)
            replacement = rng.choice(synonyms)
            # preserve capitalisation
            if word[0].isupper():
                replacement = replacement.capitalize()
            new_words = words.copy()
            new_words[idx] = replacement
            results.append(
                PerturbationResult(
                    original=prompt,
                    perturbed=" ".join(new_words),
                    strategy=self.name,
                    token_index=idx,
                    metadata={"original_word": word, "replacement": replacement},
                )
            )
        return results


class WordDeletion(Perturbation):
    """Remove individual non-stopword tokens."""

    _STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "and", "or", "but", "if", "or", "because", "while",
    }

    @property
    def name(self) -> str:
        return "word_deletion"

    def apply(self, prompt: str, n: int = 5, seed: Optional[int] = None) -> List[PerturbationResult]:
        rng = random.Random(seed)
        words = prompt.split()
        candidates = [
            i for i, w in enumerate(words)
            if w.lower().strip(string.punctuation) not in self._STOPWORDS
        ]

        results = []
        seen: set[int] = set()
        attempts = 0
        while len(results) < n and attempts < n * 3:
            attempts += 1
            if not candidates:
                break
            idx = rng.choice(candidates)
            if idx in seen:
                continue
            seen.add(idx)
            new_words = [w for i, w in enumerate(words) if i != idx]
            results.append(
                PerturbationResult(
                    original=prompt,
                    perturbed=" ".join(new_words),
                    strategy=self.name,
                    token_index=idx,
                    metadata={"deleted_word": words[idx]},
                )
            )
        return results


class Paraphrase(Perturbation):
    """
    Structural paraphrases via simple template rewrites.

    Works without any external model — suitable for quick local analysis.
    For richer paraphrasing, combine with an LLM-backed perturbation.
    """

    _TRANSFORMS = [
        # imperative → question
        (r"^(Please\s+|Kindly\s+)?(\w.*)\.$", r"Can you \2?"),
        (r"^(\w.*)\.$", r"Could you \1?"),
        # active → passive-ish
        (r"^Write (.+)$", r"Produce \1"),
        (r"^List (.+)$", r"Enumerate \1"),
        (r"^Explain (.+)$", r"Provide an explanation of \1"),
        (r"^Summarize (.+)$", r"Give a summary of \1"),
        (r"^Analyze (.+)$", r"Perform an analysis of \1"),
        (r"^Find (.+)$", r"Identify \1"),
        (r"^Compare (.+)$", r"Provide a comparison of \1"),
        (r"^Generate (.+)$", r"Produce \1"),
    ]

    @property
    def name(self) -> str:
        return "paraphrase"

    def apply(self, prompt: str, n: int = 5, seed: Optional[int] = None) -> List[PerturbationResult]:
        results = []
        for pattern, replacement in self._TRANSFORMS:
            new_prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
            if new_prompt != prompt and new_prompt not in {r.perturbed for r in results}:
                results.append(
                    PerturbationResult(
                        original=prompt,
                        perturbed=new_prompt,
                        strategy=self.name,
                        metadata={"pattern": pattern},
                    )
                )
            if len(results) >= n:
                break
        return results


class CasingVariation(Perturbation):
    """Varies casing of key words (title, upper, lower)."""

    @property
    def name(self) -> str:
        return "casing_variation"

    def apply(self, prompt: str, n: int = 5, seed: Optional[int] = None) -> List[PerturbationResult]:
        variants = [
            prompt.lower(),
            prompt.upper(),
            prompt.title(),
            " ".join(
                w.upper() if i % 3 == 0 else w.lower()
                for i, w in enumerate(prompt.split())
            ),
        ]
        return [
            PerturbationResult(original=prompt, perturbed=v, strategy=self.name)
            for v in variants[:n]
            if v != prompt
        ]


class PunctuationVariation(Perturbation):
    """Adds, removes, or changes terminal punctuation."""

    @property
    def name(self) -> str:
        return "punctuation_variation"

    def apply(self, prompt: str, n: int = 5, seed: Optional[int] = None) -> List[PerturbationResult]:
        stripped = prompt.rstrip(".!?")
        variants = [
            stripped,
            stripped + ".",
            stripped + "!",
            stripped + "?",
            stripped + "...",
        ]
        return [
            PerturbationResult(original=prompt, perturbed=v, strategy=self.name)
            for v in variants[:n]
            if v != prompt
        ]


class OrderShuffle(Perturbation):
    """Shuffles sentence order within a multi-sentence prompt."""

    @property
    def name(self) -> str:
        return "order_shuffle"

    def apply(self, prompt: str, n: int = 5, seed: Optional[int] = None) -> List[PerturbationResult]:
        rng = random.Random(seed)
        sentences = re.split(r"(?<=[.!?])\s+", prompt.strip())
        if len(sentences) < 2:
            return []

        results = []
        seen: set[str] = {prompt}
        for _ in range(n * 3):
            shuffled = sentences.copy()
            rng.shuffle(shuffled)
            candidate = " ".join(shuffled)
            if candidate not in seen:
                seen.add(candidate)
                results.append(
                    PerturbationResult(original=prompt, perturbed=candidate, strategy=self.name)
                )
            if len(results) >= n:
                break
        return results


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_PERTURBATIONS: list[Perturbation] = [
    SynonymSubstitution(),
    WordDeletion(),
    Paraphrase(),
    CasingVariation(),
    PunctuationVariation(),
    OrderShuffle(),
]

PERTURBATION_REGISTRY: dict[str, Perturbation] = {p.name: p for p in ALL_PERTURBATIONS}
