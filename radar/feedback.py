"""Ratings, saves and clicks, shared by the web app and the email rating links."""

import base64
import hashlib
import hmac
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from radar.db.models import Feedback, ProfilePaper

RATINGS = {"up", "down", "clear"}
SAVES = {"save", "unsave"}
CLICKS = {"click_abs", "click_pdf"}


@dataclass
class PaperState:
    rating: str | None = None  # 'up' | 'down' | None
    saved: bool = False


def record(
    session: Session,
    user_id: int,
    arxiv_id: str,
    action: str,
    feed_id: int | None = None,
    via: str = "web",
) -> None:
    if action not in RATINGS | SAVES | CLICKS:
        raise ValueError(f"unknown action {action!r}")
    session.add(
        Feedback(user_id=user_id, arxiv_id=arxiv_id, feed_id=feed_id, action=action, via=via)
    )
    if action in ("up", "save"):
        source = "liked" if action == "up" else "saved"
        session.execute(
            insert(ProfilePaper)
            .values(user_id=user_id, arxiv_id=arxiv_id, source=source)
            .on_conflict_do_nothing()
        )
    elif action in ("down", "clear"):
        _remove_from_profile(session, user_id, arxiv_id, "liked")
    elif action == "unsave":
        _remove_from_profile(session, user_id, arxiv_id, "saved")
    session.flush()


def _remove_from_profile(session: Session, user_id: int, arxiv_id: str, source: str) -> None:
    """Undo a like or save, but never remove authored or seed papers."""
    session.execute(
        delete(ProfilePaper).where(
            ProfilePaper.user_id == user_id,
            ProfilePaper.arxiv_id == arxiv_id,
            ProfilePaper.source == source,
        )
    )


def states(session: Session, user_id: int, arxiv_ids: list[str]) -> dict[str, PaperState]:
    out = {a: PaperState() for a in arxiv_ids}
    events = session.execute(
        select(Feedback.arxiv_id, Feedback.action)
        .where(Feedback.user_id == user_id, Feedback.arxiv_id.in_(arxiv_ids))
        .order_by(Feedback.at, Feedback.id)
    ).all()
    for arxiv_id, action in events:
        state = out[arxiv_id]
        if action in ("up", "down"):
            state.rating = action
        elif action == "clear":
            state.rating = None
        elif action in SAVES:
            state.saved = action == "save"
    return out


# Signed tokens let email links rate papers without a login and without forgery.


def sign(secret: str, user_id: int, arxiv_id: str, action: str, feed_id: int | None) -> str:
    payload = f"{user_id}|{arxiv_id}|{action}|{feed_id or ''}".encode()
    mac = hmac.new(secret.encode(), payload, hashlib.sha256).digest()[:16]
    return f"{_b64(payload)}.{_b64(mac)}"


def verify(secret: str, token: str) -> tuple[int, str, str, int | None] | None:
    try:
        payload_b64, mac_b64 = token.split(".")
        payload = _unb64(payload_b64)
        expected = hmac.new(secret.encode(), payload, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(expected, _unb64(mac_b64)):
            return None
        user_id, arxiv_id, action, feed_id = payload.decode().split("|")
        return int(user_id), arxiv_id, action, int(feed_id) if feed_id else None
    except (ValueError, UnicodeDecodeError):
        return None


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
