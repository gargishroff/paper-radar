"""The daily email digest and failure alerts."""

import logging
import smtplib
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import Engine, select

from radar.config import Settings
from radar.db import session_scope
from radar.db.models import Feed, FeedItem, Interest, Paper, User
from radar.feedback import sign

log = logging.getLogger(__name__)

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    autoescape=select_autoescape(["html"]),
)


def render_digest(settings: Settings, user: User, feed: Feed, rows) -> tuple[str, str, str]:
    """rows: (FeedItem, Paper, interest name | None). Returns (subject, html, text)."""
    base = settings.app_base_url.rstrip("/")

    def link(arxiv_id: str, action: str) -> str:
        return f"{base}/r/{sign(settings.secret_key, user.id, arxiv_id, action, feed.id)}"

    items = [
        {
            "paper": paper,
            "item": item,
            "interest": interest,
            "open": f"{base}/go/{paper.arxiv_id}?to=abs&feed_id={feed.id}",
            "pdf": f"{base}/go/{paper.arxiv_id}?to=pdf&feed_id={feed.id}",
            "up": link(paper.arxiv_id, "up"),
            "down": link(paper.arxiv_id, "down"),
            "save": link(paper.arxiv_id, "save"),
        }
        for item, paper, interest in rows
    ]
    subject = f"PaperRadar · {feed.feed_date.strftime('%a %d %b')} · {len(items)} papers"
    ctx = {"items": items, "feed": feed, "base": base, "user": user}
    html = _env.get_template("digest.html").render(**ctx)
    text = _env.get_template("digest.txt").render(**ctx)
    return subject, html, text


def deliver(settings: Settings, to: str, subject: str, html: str | None, text: str) -> str:
    """Send by SMTP, or write to `digest_dir` without SMTP; returns the destination."""
    if not settings.smtp_host:
        out = Path(settings.digest_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        path = out / f"{stamp}-{subject.split('·')[1].strip().replace(' ', '-')}.html"
        path.write_text(html or f"<pre>{text}</pre>", encoding="utf-8")
        return str(path)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.smtp_from or settings.smtp_user
    msg["To"] = to
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
        smtp.starttls()
        if settings.smtp_user:
            smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(msg)
    return to


def send_digest(
    engine: Engine, settings: Settings, feed_id: int, force: bool = False
) -> str | None:
    """Email a feed once. Returns the destination, or None if skipped."""
    with session_scope(engine) as session:
        feed = session.get(Feed, feed_id)
        user = session.get(User, feed.user_id)
        if (feed.stats or {}).get("emailed_at") and not force:
            return None
        if not user.email and settings.smtp_host:
            log.info("user %s has no email; digest skipped", user.id)
            return None
        rows = session.execute(
            select(FeedItem, Paper, Interest.name)
            .join(Paper, Paper.arxiv_id == FeedItem.arxiv_id)
            .outerjoin(Interest, Interest.id == FeedItem.interest_id)
            .where(FeedItem.feed_id == feed_id)
            .order_by(FeedItem.rank)
        ).all()
        subject, html, text = render_digest(settings, user, feed, rows)
        dest = deliver(settings, user.email or "", subject, html, text)
        feed.stats = {**(feed.stats or {}), "emailed_at": datetime.now(UTC).isoformat()}
        log.info("digest for feed %s -> %s", feed_id, dest)
        return dest


def send_alert(settings: Settings, to: str | None, subject: str, body: str) -> None:
    """Best-effort failure alert; never raises, since it runs inside error handling."""
    try:
        deliver(settings, to or settings.smtp_from, f"[PaperRadar] · {subject}", None, body)
    except Exception:
        log.exception("could not send alert")
