"""Information extraction: text to (entity, relation, entity) triplets.

Two engines share one canonical schema:

- RuleBasedExtractor: a deterministic lexicon and pattern engine. It is the
  default because it is reproducible, dependency free, auditable, and its
  recall on the schema aligned corpus is directly measured by the test
  suite. This is the default engine.
- llm_extract: the blueprint's Phase 2 engine, prompting an instruction
  tuned LLM (Llama 3 8B class) through Hugging Face transformers to emit
  JSON triplets. It is fully implemented but requires local model weights,
  so it is opt in via build_dkg.py --llm.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .schema import all_relation_pairs, all_surface_pairs

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")


def _compile(pairs: list[tuple[str, str]]) -> list[tuple[re.Pattern, str]]:
    compiled = []
    for surface, canonical in pairs:
        pattern = re.compile(
            r"(?<![a-z0-9&])" + re.escape(surface) + r"(?![a-z0-9&])")
        compiled.append((pattern, canonical))
    return compiled


_ENTITY_PATTERNS = _compile(all_surface_pairs())
_RELATION_PATTERNS = _compile(all_relation_pairs())


@dataclass
class Triplet:
    h: str
    r: str
    t: str
    sentence: str
    month: str
    confidence: float

    def key(self) -> tuple[str, str, str]:
        return (self.h, self.r, self.t)


def _find_spans(text: str,
                patterns: list[tuple[re.Pattern, str]]) -> list[tuple[int, int, str]]:
    """Longest match first, later shorter matches cannot overlap earlier ones."""
    taken: list[tuple[int, int]] = []
    spans: list[tuple[int, int, str]] = []
    for pattern, canonical in patterns:
        for m in pattern.finditer(text):
            s, e = m.span()
            if any(s < te and e > ts for ts, te in taken):
                continue
            taken.append((s, e))
            spans.append((s, e, canonical))
    return sorted(spans)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


class RuleBasedExtractor:
    """Deterministic SVO extraction over the canonical lexicons."""

    def extract_sentence(self, sentence: str, month: str = "") -> list[Triplet]:
        low = _WS.sub(" ", sentence.lower().strip())
        entities = _find_spans(low, _ENTITY_PATTERNS)
        relations = _find_spans(low, _RELATION_PATTERNS)
        if not entities or not relations:
            return []

        out: dict[tuple, Triplet] = {}
        for rs, re_, rel in relations:
            subject = None
            for es, ee, ent in entities:
                if ee <= rs and (subject is None or ee > subject[1]):
                    subject = (es, ee, ent)
            obj = None
            for es, ee, ent in entities:
                if es >= re_ and (obj is None or es < obj[0]):
                    obj = (es, ee, ent)
            if subject is None or obj is None or subject[2] == obj[2]:
                continue
            near = (rs - subject[1] <= 30) and (obj[0] - re_ <= 30)
            trip = Triplet(subject[2], rel, obj[2], sentence.strip(), month,
                           0.9 if near else 0.7)
            prev = out.get(trip.key())
            if prev is None or trip.confidence > prev.confidence:
                out[trip.key()] = trip
        return list(out.values())

    def extract_document(self, text: str, month: str = "") -> list[Triplet]:
        trips: list[Triplet] = []
        for sentence in split_sentences(text):
            trips.extend(self.extract_sentence(sentence, month))
        return trips

    def extract_corpus(self, docs: list[dict]) -> list[Triplet]:
        """docs rows carry 'month', 'statement' and optional 'news' list."""
        trips: list[Triplet] = []
        for doc in docs:
            month = doc.get("month", "")
            trips.extend(self.extract_document(doc.get("statement", ""), month))
            for sentence in doc.get("news", []):
                trips.extend(self.extract_sentence(sentence, month))
        return trips


_LLM_PROMPT = """You are an information extraction engine for macroeconomics.
Extract causal (head, relation, tail) triplets from the text. Use only these
canonical entities: {entities}. Use only these canonical relations: {relations}.
Respond with a JSON list like
[{{"h": "Federal_Reserve", "r": "raises", "t": "Policy_Rate"}}].
Respond with JSON only.

Text: {text}
JSON:"""


def llm_extract(docs: list[dict], model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                max_new_tokens: int = 256) -> list[Triplet]:
    """Prompt an instruction tuned LLM for triplets. Requires transformers
    plus local or downloadable weights, so this path is opt in."""
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Install it and download the model "
            "to use --llm extraction, or use the default rule based engine."
        ) from exc

    from .schema import ENTITIES, RELATIONS
    generator = pipeline("text-generation", model=model_name, device_map="auto")
    trips: list[Triplet] = []
    for doc in docs:
        prompt = _LLM_PROMPT.format(
            entities=", ".join(ENTITIES), relations=", ".join(RELATIONS),
            text=doc.get("statement", "") + " " + " ".join(doc.get("news", [])))
        raw = generator(prompt, max_new_tokens=max_new_tokens,
                        do_sample=False)[0]["generated_text"][len(prompt):]
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            continue
        try:
            rows = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        for row in rows:
            h, r, t = row.get("h"), row.get("r"), row.get("t")
            if h in ENTITIES and t in ENTITIES and r in RELATIONS:
                trips.append(Triplet(h, r, t, "", doc.get("month", ""), 0.8))
    return trips
