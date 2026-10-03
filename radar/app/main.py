"""PaperRadar web app: Today, Interests, Library, Search, Onboarding, Status."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Engine, delete, func, select, text

from radar import feedback
from radar.config import Settings, get_settings
from radar.db import make_engine, session_scope
from radar.db.models import (
    ApiUsage,
    Feed,
    Feedback,
    FeedItem,
    Interest,
    JobRun,
    Paper,
    PaperEmbedding,
    ProfilePaper,
    User,
)
from radar.embed.pipeline import normalize
from radar.profile.build import (
    add_profile_papers,
    add_seed_interest,
    add_text_interest,
    query_text,
    rebuild_interests,
)
from radar.rank.feed import build_feed
from radar.services import Services, make_services
from radar.usage import BudgetExceeded

log = logging.getLogger(__name__)
HERE = Path(__file__).parent
ACTIONS = {"up", "down", "clear", "save", "unsave"}


@dataclass
class Card:
    paper: Paper
    state: feedback.PaperState
    feed_id: int | None = None
    item: FeedItem | None = None
    interest: str | None = None


def create_app(
    engine: Engine | None = None,
    settings: Settings | None = None,
    services: Callable[[], Services] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    engine = engine or make_engine(settings.database_url)
    _services: dict[str, Services] = {}

    def svc() -> Services:
        # Built lazily so the app starts (and the read-only pages work) without an API key.
        if "s" not in _services:
            _services["s"] = services() if services else make_services(engine, settings)
        return _services["s"]

    app = FastAPI(title="PaperRadar", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")

    def render(request: Request, name: str, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, name, ctx)

    def current_user(session) -> User | None:
        return session.scalar(select(User).order_by(User.id))

    def cards_for(session, user_id: int, rows) -> list[Card]:
        """rows: (Paper, FeedItem | None, feed_id | None)"""
        ids = [r[0].arxiv_id for r in rows]
        state = feedback.states(session, user_id, ids)
        names = dict(session.execute(select(Interest.id, Interest.name)).all())
        return [
            Card(
                paper=p,
                state=state[p.arxiv_id],
                feed_id=feed_id,
                item=item,
                interest=names.get(item.interest_id) if item and item.interest_id else None,
            )
            for p, item, feed_id in rows
        ]

    # ---- Today -------------------------------------------------------------------------

    @app.get("/")
    def index() -> RedirectResponse:
        return RedirectResponse("/today", status_code=303)

    @app.get("/today", response_class=HTMLResponse)
    def today(request: Request, day: date | None = None):
        with session_scope(engine) as session:
            user = current_user(session)
            if user is None:
                return RedirectResponse("/onboarding", status_code=303)
            dates = session.scalars(
                select(Feed.feed_date).where(Feed.user_id == user.id).order_by(Feed.feed_date)
            ).all()
            day = day or (dates[-1] if dates else None)
            feed = (
                session.scalar(select(Feed).where(Feed.user_id == user.id, Feed.feed_date == day))
                if day
                else None
            )
            cards: list[Card] = []
            if feed:
                rows = session.execute(
                    select(Paper, FeedItem)
                    .join(FeedItem, FeedItem.arxiv_id == Paper.arxiv_id)
                    .where(FeedItem.feed_id == feed.id)
                    .order_by(FeedItem.rank)
                ).all()
                cards = cards_for(session, user.id, [(p, i, feed.id) for p, i in rows])
            earlier = [d for d in dates if day and d < day]
            later = [d for d in dates if day and d > day]
            latest_papers = session.scalar(select(func.max(Paper.announced_on)))
            return render(
                request,
                "today.html",
                cards=cards,
                feed=feed,
                day=day,
                prev_day=earlier[-1] if earlier else None,
                next_day=later[0] if later else None,
                latest_papers=latest_papers,
            )

    @app.post("/feed/build")
    def feed_build(day: Annotated[date, Form()]):
        with session_scope(engine) as session:
            user = current_user(session)
        s = svc()
        build_feed(engine, settings, user.id, day, s.embedder.model, s.reranker, s.chat)
        return RedirectResponse(f"/today?day={day}", status_code=303)

    # ---- Feedback ----------------------------------------------------------------------

    @app.post("/rate/{arxiv_id}/{action}", response_class=HTMLResponse)
    def rate(request: Request, arxiv_id: str, action: str, feed_id: int | None = None):
        if action not in ACTIONS:
            raise HTTPException(400, "unknown action")
        with session_scope(engine) as session:
            user = current_user(session)
            if session.get(Paper, arxiv_id) is None:
                raise HTTPException(404)
            feedback.record(session, user.id, arxiv_id, action, feed_id)
            state = feedback.states(session, user.id, [arxiv_id])[arxiv_id]
        return render(request, "_rating.html", arxiv_id=arxiv_id, state=state, feed_id=feed_id)

    @app.get("/go/{arxiv_id}")
    def go(arxiv_id: str, to: str = "abs", feed_id: int | None = None):
        with session_scope(engine) as session:
            user = current_user(session)
            if user and session.get(Paper, arxiv_id):
                feedback.record(
                    session, user.id, arxiv_id, "click_pdf" if to == "pdf" else "click_abs", feed_id
                )
        kind = "pdf" if to == "pdf" else "abs"
        return RedirectResponse(f"https://arxiv.org/{kind}/{arxiv_id}", status_code=302)

    @app.get("/r/{token}", response_class=HTMLResponse)
    def email_rating(request: Request, token: str):
        parsed = feedback.verify(settings.secret_key, token)
        if parsed is None:
            raise HTTPException(400, "invalid link")
        user_id, arxiv_id, action, feed_id = parsed
        if action not in ACTIONS:
            raise HTTPException(400, "invalid link")
        with session_scope(engine) as session:
            feedback.record(session, user_id, arxiv_id, action, feed_id, via="email")
            paper = session.get(Paper, arxiv_id)
        return render(request, "rated.html", paper=paper, action=action)

    # ---- Interests ---------------------------------------------------------------------

    @app.get("/interests", response_class=HTMLResponse)
    def interests(request: Request, error: str | None = None):
        with session_scope(engine) as session:
            user = current_user(session)
            if user is None:
                return RedirectResponse("/onboarding", status_code=303)
            rows = []
            for i in session.scalars(
                select(Interest).where(Interest.user_id == user.id).order_by(Interest.weight.desc())
            ):
                papers = session.scalars(
                    select(Paper)
                    .join(ProfilePaper, ProfilePaper.arxiv_id == Paper.arxiv_id)
                    .where(ProfilePaper.interest_id == i.id)
                    .order_by(Paper.published_at.desc())
                ).all()
                rows.append((i, papers))
            return render(request, "interests.html", interests=rows, error=error)

    @app.post("/interests/{interest_id}/edit")
    def interest_edit(
        interest_id: int,
        name: Annotated[str, Form()],
        description: Annotated[str, Form()],
    ):
        with session_scope(engine) as session:
            interest = session.get(Interest, interest_id)
            if interest is None:
                raise HTTPException(404)
            changed = (interest.name, interest.description) != (name.strip(), description.strip())
            interest.name, interest.description = name.strip(), description.strip()
            interest.pinned = True  # an edited interest survives "rebuild interests"
            needs_vector = changed and interest.medoid_arxiv_id is None
        if needs_vector:
            [vec] = svc().embedder.embed(
                [query_text(name.strip(), description.strip())], "search_query", essential=True
            )
            with session_scope(engine) as session:
                session.get(Interest, interest_id).desc_embedding = normalize(vec)
        return RedirectResponse("/interests", status_code=303)

    @app.post("/interests/{interest_id}/delete")
    def interest_delete(interest_id: int):
        with session_scope(engine) as session:
            interest = session.get(Interest, interest_id)
            if interest is None:
                raise HTTPException(404)
            # Drop its papers too, or they would be reassigned to another interest.
            session.execute(delete(ProfilePaper).where(ProfilePaper.interest_id == interest_id))
            session.delete(interest)
        return RedirectResponse("/interests", status_code=303)

    @app.post("/interests/new")
    def interest_new(
        seeds: Annotated[str, Form()] = "",
        name: Annotated[str, Form()] = "",
        description: Annotated[str, Form()] = "",
    ):
        with session_scope(engine) as session:
            user = current_user(session)
        s = svc()
        try:
            if seeds.split():
                created = add_seed_interest(
                    engine, s.fetcher, s.embedder, s.chat, settings, user.id, seeds.split()
                )
                if created is None:
                    return RedirectResponse(
                        "/interests?error=None of those arXiv IDs were found", status_code=303
                    )
            elif description.strip():
                add_text_interest(
                    engine,
                    s.embedder,
                    settings,
                    user.id,
                    name.strip() or description.strip()[:60],
                    description.strip(),
                )
        except BudgetExceeded as exc:
            return RedirectResponse(f"/interests?error={exc}", status_code=303)
        return RedirectResponse("/interests", status_code=303)

    @app.post("/interests/rebuild")
    def interest_rebuild():
        with session_scope(engine) as session:
            user = current_user(session)
        s = svc()
        rebuild_interests(engine, s.chat, settings, user.id, s.embedder.model)
        return RedirectResponse("/interests", status_code=303)

    # ---- Library -----------------------------------------------------------------------

    def saved_papers(session, user_id: int) -> list[Paper]:
        ids = session.scalars(
            select(Feedback.arxiv_id)
            .where(Feedback.user_id == user_id, Feedback.action.in_(feedback.SAVES))
            .distinct()
        ).all()
        state = feedback.states(session, user_id, list(ids))
        saved = [a for a in ids if state[a].saved]
        return session.scalars(
            select(Paper).where(Paper.arxiv_id.in_(saved)).order_by(Paper.published_at.desc())
        ).all()

    @app.get("/library", response_class=HTMLResponse)
    def library(request: Request):
        with session_scope(engine) as session:
            user = current_user(session)
            if user is None:
                return RedirectResponse("/onboarding", status_code=303)
            papers = saved_papers(session, user.id)
            cards = cards_for(session, user.id, [(p, None, None) for p in papers])
            return render(request, "library.html", cards=cards)

    @app.get("/library.bib", response_class=PlainTextResponse)
    def library_bib():
        with session_scope(engine) as session:
            user = current_user(session)
            papers = saved_papers(session, user.id) if user else []
            return "\n".join(bibtex(p) for p in papers)

    # ---- Search ------------------------------------------------------------------------

    @app.get("/search", response_class=HTMLResponse)
    def search(request: Request, q: str = ""):
        cards: list[Card] = []
        error = None
        if q.strip():
            s = svc()
            try:
                [vec] = s.embedder.embed([q.strip()], "search_query")
            except BudgetExceeded as exc:
                error = str(exc)
            else:
                with session_scope(engine) as session:
                    user = current_user(session)
                    if user is None:
                        return RedirectResponse("/onboarding", status_code=303)
                    distance = PaperEmbedding.embedding.cosine_distance(normalize(vec))
                    papers = session.scalars(
                        select(Paper)
                        .join(PaperEmbedding, PaperEmbedding.arxiv_id == Paper.arxiv_id)
                        .where(PaperEmbedding.model == s.embedder.model)
                        .order_by(distance)
                        .limit(20)
                    ).all()
                    cards = cards_for(session, user.id, [(p, None, None) for p in papers])
        return render(request, "search.html", q=q, cards=cards, error=error)

    # ---- Onboarding --------------------------------------------------------------------

    @app.get("/onboarding", response_class=HTMLResponse)
    def onboarding(request: Request, error: str | None = None):
        with session_scope(engine) as session:
            if current_user(session) is not None:
                return RedirectResponse("/interests", status_code=303)
        return render(request, "onboarding.html", error=error)

    @app.post("/onboarding")
    def onboarding_submit(
        name: Annotated[str, Form()],
        email: Annotated[str, Form()] = "",
        s2: Annotated[str, Form()] = "",
        seeds: Annotated[str, Form()] = "",
        topic: Annotated[str, Form()] = "",
    ):
        if not (s2.strip() or seeds.split() or topic.strip()):
            return RedirectResponse(
                "/onboarding?error=Give at least one of: author ID, seed papers, or a topic",
                status_code=303,
            )
        s = svc()
        with session_scope(engine) as session:
            user = User(name=name.strip(), email=email.strip() or None, s2_author_id=s2.strip())
            session.add(user)
            session.flush()
            uid = user.id
        try:
            if s2.strip():
                ids = s.fetcher.author_arxiv_ids(s2.strip())
                add_profile_papers(engine, s.fetcher, s.embedder, uid, ids, "authored")
                rebuild_interests(engine, s.chat, settings, uid, s.embedder.model)
            if seeds.split():
                add_seed_interest(
                    engine, s.fetcher, s.embedder, s.chat, settings, uid, seeds.split()
                )
            if topic.strip():
                topic_name, _, description = topic.partition(":")
                add_text_interest(
                    engine,
                    s.embedder,
                    settings,
                    uid,
                    topic_name.strip()[:60],
                    (description or topic).strip(),
                )
        except Exception as exc:
            # Leave no half-built user behind: the next attempt starts clean.
            log.exception("onboarding failed")
            with session_scope(engine) as session:
                session.execute(delete(User).where(User.id == uid))
            return RedirectResponse(
                f"/onboarding?error=Setup failed ({type(exc).__name__}: {exc}). Please retry.",
                status_code=303,
            )
        # Land on a real feed straight away when papers are already ingested.
        with session_scope(engine) as session:
            latest = session.scalar(select(func.max(Paper.announced_on)))
        if latest is not None:
            build_feed(engine, settings, uid, latest, s.embedder.model, s.reranker, s.chat)
            return RedirectResponse("/today", status_code=303)
        return RedirectResponse("/interests", status_code=303)

    # ---- Status ------------------------------------------------------------------------

    @app.get("/status", response_class=HTMLResponse)
    def status(request: Request):
        month_start = datetime.now(UTC).date().replace(day=1)
        with session_scope(engine) as session:
            runs = session.scalars(select(JobRun).order_by(JobRun.id.desc()).limit(15)).all()
            usage = session.execute(
                select(ApiUsage.endpoint, func.sum(ApiUsage.calls), func.sum(ApiUsage.units))
                .where(ApiUsage.day >= month_start)
                .group_by(ApiUsage.endpoint)
                .order_by(ApiUsage.endpoint)
            ).all()
            n_papers = session.scalar(select(func.count()).select_from(Paper))
            n_vectors = session.scalar(select(func.count()).select_from(PaperEmbedding))
            last_feed = session.scalar(select(Feed).order_by(Feed.feed_date.desc()))
            return render(
                request,
                "status.html",
                runs=runs,
                usage=usage,
                calls=sum(u[1] for u in usage),
                budget=settings.monthly_call_budget,
                n_papers=n_papers,
                n_vectors=n_vectors,
                last_feed=last_feed,
            )

    @app.get("/healthz")
    def healthz() -> Response:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return PlainTextResponse("ok")

    return app


def bibtex(p: Paper) -> str:
    key = p.arxiv_id.replace("/", "_").replace(".", "_")
    authors = " and ".join(p.authors)
    return (
        f"@misc{{arxiv_{key},\n"
        f"  title = {{{p.title}}},\n"
        f"  author = {{{authors}}},\n"
        f"  year = {{{p.published_at.year}}},\n"
        f"  eprint = {{{p.arxiv_id}}},\n"
        f"  archivePrefix = {{arXiv}},\n"
        f"  primaryClass = {{{p.primary_cat}}},\n"
        f"  url = {{https://arxiv.org/abs/{p.arxiv_id}}}\n"
        f"}}\n"
    )


def __getattr__(name: str):
    # Built lazily so importing this module needs no database.
    if name == "app":
        return create_app()
    raise AttributeError(name)
