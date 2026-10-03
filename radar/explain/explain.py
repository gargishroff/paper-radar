"""TL;DRs and "why this is relevant to you" lines, one Command A call per feed."""

import logging
import re
from dataclasses import dataclass

from radar.llm import ChatModel
from radar.usage import BudgetExceeded

log = logging.getLogger(__name__)

ABSTRACT_CHARS = 1200


@dataclass(frozen=True)
class ExplainInput:
    arxiv_id: str
    title: str
    abstract: str
    neighbours: list[tuple[str, str]]  # (arxiv_id, title) of the most similar profile papers
    interest: tuple[str, str] | None = None  # (name, description) of the matched interest
    explore: bool = False


@dataclass(frozen=True)
class Explanation:
    tldr: str
    why: str
    grounded: bool  # False when the fallback text was used


_SCHEMA = {
    "type": "object",
    "properties": {
        "papers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "tldr": {"type": "string"},
                    "why": {"type": "string"},
                    "relates_to": {"type": "string"},
                },
                "required": ["arxiv_id", "tldr", "why", "relates_to"],
            },
        }
    },
    "required": ["papers"],
}

INTEREST_REF = "interest"
EXPLORE_WHY = (
    "Outside your usual interests: shown so PaperRadar learns whether this area is useful."
)

_SYSTEM = (
    "You write a researcher's daily paper digest, addressed to them as 'you'. For each new "
    "paper: 'tldr' is one plain sentence (max 30 words) saying what the paper does and finds. "
    "'why' is one sentence (max 30 words) that connects the paper to the reader: name their "
    "interest or one of their listed papers, then say concretely what this paper offers for it "
    "(a method, result, or benchmark they could use or compare against). Do not repeat the "
    "tldr. If the link is loose, say so plainly. Never write 'our' or 'we'. 'relates_to' is "
    "the arXiv ID of the reader's paper you named, copied exactly, or the word 'interest' if "
    "you named the interest. Do not invent results that are not in the abstract."
)


def first_sentence(text: str, limit: int = 200) -> str:
    sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "…"


def fallback(item: ExplainInput) -> Explanation:
    if item.explore:
        why = EXPLORE_WHY
    elif item.neighbours:
        why = f"Similar to: {item.neighbours[0][1]}"
    elif item.interest:
        why = f"Matches your interest: {item.interest[0]}"
    else:
        why = ""
    return Explanation(first_sentence(item.abstract), why, grounded=False)


def explain(chat: ChatModel | None, items: list[ExplainInput]) -> dict[str, Explanation]:
    out = {item.arxiv_id: fallback(item) for item in items}
    if chat is None or not items:
        return out

    blocks = []
    for item in items:
        if item.explore:
            blocks.append(
                f"NEW PAPER [{item.arxiv_id}]: {item.title}\n"
                f"Abstract: {item.abstract[:ABSTRACT_CHARS]}\n"
                "  (Exploration pick: write only the tldr; set why to '' and relates_to to "
                "'interest'.)"
            )
            continue
        mine = "\n".join(f"    - [{a}] {t}" for a, t in item.neighbours) or "    (none)"
        name, description = item.interest or ("(unspecified)", "")
        blocks.append(
            f"NEW PAPER [{item.arxiv_id}]: {item.title}\n"
            f"Abstract: {item.abstract[:ABSTRACT_CHARS]}\n"
            f"  Reader's interest: {name}. {description}\n"
            f"  Reader's papers in this interest:\n{mine}"
        )
    try:
        reply = chat.chat_json(_SYSTEM, "\n\n".join(blocks), _SCHEMA)
    except BudgetExceeded as exc:
        log.warning("explanations skipped: %s", exc)
        return out

    by_id = {item.arxiv_id: item for item in items}
    for entry in reply.get("papers", []):
        item = by_id.get(str(entry.get("arxiv_id", "")).strip())
        if item is None:
            continue
        if item.explore:
            tldr = (entry.get("tldr") or "").strip() or out[item.arxiv_id].tldr
            out[item.arxiv_id] = Explanation(tldr, EXPLORE_WHY, grounded=False)
            continue
        allowed = {a for a, _ in item.neighbours} | ({INTEREST_REF} if item.interest else set())
        tldr = (entry.get("tldr") or "").strip() or out[item.arxiv_id].tldr
        cited = str(entry.get("relates_to", "")).strip()
        if cited in allowed and (entry.get("why") or "").strip():
            out[item.arxiv_id] = Explanation(tldr, entry["why"].strip(), grounded=True)
        else:
            out[item.arxiv_id] = Explanation(tldr, out[item.arxiv_id].why, grounded=False)
    return out
