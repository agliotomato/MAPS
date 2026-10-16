from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.forecast.price_forecast import backtest_price_forecasts


def _to_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _build_windows(start_date: date, end_date: date, window_days: int) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cur = start_date
    while cur <= end_date:
        nxt = min(end_date, cur + timedelta(days=max(1, window_days) - 1))
        windows.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return windows


def _task_key(company_code: str, start_date: date, end_date: date, step_days: int) -> str:
    return f"{company_code}|{start_date.isoformat()}|{end_date.isoformat()}|{step_days}"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed": {}, "history": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"completed": {}, "history": []}


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_chunked_semiconductor_backtests(
    *,
    company_codes: list[str],
    start_date: date | str,
    end_date: date | str,
    horizons: list[int],
    step_days: int = 21,
    window_days: int = 180,
    max_tasks: int = 3,
    persist: bool = True,
    checkpoint_path: str | Path = "reports/overnight_backtest_checkpoint.json",
) -> dict[str, Any]:
    start_dt = _to_date(start_date)
    end_dt = _to_date(end_date)
    path = Path(checkpoint_path)
    checkpoint = _load_checkpoint(path)
    completed = checkpoint.setdefault("completed", {})
    history = checkpoint.setdefault("history", [])

    tasks: list[dict[str, Any]] = []
    for company_code in company_codes:
        for win_start, win_end in _build_windows(start_dt, end_dt, window_days):
            key = _task_key(company_code, win_start, win_end, step_days)
            tasks.append(
                {
                    "key": key,
                    "company_code": company_code,
                    "start_date": win_start.isoformat(),
                    "end_date": win_end.isoformat(),
                    "horizons": list(horizons),
                    "step_days": int(step_days),
                    "done": bool(completed.get(key)),
                }
            )

    pending = [t for t in tasks if not t["done"]]
    ran: list[dict[str, Any]] = []
    total_cases = 0

    for task in pending[: max(1, int(max_tasks))]:
        result = backtest_price_forecasts(
            company_codes=[str(task["company_code"])],
            horizons=list(task["horizons"]),
            start_date=str(task["start_date"]),
            end_date=str(task["end_date"]),
            step_days=int(task["step_days"]),
            persist=persist,
        )
        summary = {
            "key": task["key"],
            "company_code": task["company_code"],
            "start_date": task["start_date"],
            "end_date": task["end_date"],
            "cases": int(result.get("cases", 0) or 0),
            "interval_hit_rate": float(result.get("interval_hit_rate", 0.0) or 0.0),
            "direction_accuracy": float(result.get("direction_accuracy", 0.0) or 0.0),
            "avg_abs_error": float(result.get("avg_abs_error", 0.0) or 0.0),
        }
        completed[task["key"]] = summary
        history.append(summary)
        _save_checkpoint(path, checkpoint)
        ran.append(summary)
        total_cases += summary["cases"]

    return {
        "total_tasks": len(tasks),
        "completed_tasks": len(completed),
        "pending_tasks": max(0, len(tasks) - len(completed)),
        "ran_tasks": ran,
        "ran_cases": total_cases,
        "checkpoint_path": str(path),
    }
