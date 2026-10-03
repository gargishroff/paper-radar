"""Cohere call accounting and the monthly budget guard."""

from datetime import UTC, date, datetime

from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert

from radar.config import Settings
from radar.db import session_scope
from radar.db.models import ApiUsage


class BudgetExceeded(RuntimeError):
    pass


class UsageTracker:
    def __init__(self, engine: Engine, settings: Settings, today: date | None = None):
        self._engine = engine
        self._budget = settings.monthly_call_budget
        self._soft_limit = int(settings.monthly_call_budget * settings.nonessential_budget_fraction)
        self._today = today or datetime.now(UTC).date()

    def calls_this_month(self) -> int:
        month_start = self._today.replace(day=1)
        with session_scope(self._engine) as session:
            total = session.scalar(
                select(func.coalesce(func.sum(ApiUsage.calls), 0)).where(
                    ApiUsage.day >= month_start
                )
            )
        return int(total)

    def check(self, calls: int = 1, essential: bool = False) -> None:
        """Raise BudgetExceeded if making `calls` more calls would break the budget."""
        used = self.calls_this_month()
        limit = self._budget if essential else self._soft_limit
        if used + calls > limit:
            kind = "essential" if essential else "non-essential"
            raise BudgetExceeded(
                f"{kind} call refused: {used} of {self._budget} monthly calls used "
                f"(limit for {kind} calls: {limit})"
            )

    def record(self, endpoint: str, model: str, calls: int = 1, units: int = 0) -> None:
        stmt = insert(ApiUsage).values(
            day=self._today, endpoint=endpoint, model=model, calls=calls, units=units
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["day", "endpoint", "model"],
            set_={
                "calls": ApiUsage.calls + stmt.excluded.calls,
                "units": ApiUsage.units + stmt.excluded.units,
            },
        )
        with session_scope(self._engine) as session:
            session.execute(stmt)
