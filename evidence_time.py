"""Filter evidence before model input without inventing publication times."""
from copy import deepcopy
from datetime import date, datetime, time, timedelta
import re
from zoneinfo import ZoneInfo


EXCLUSION_REASONS = {
    "published_after_cutoff": "질문 기준 이후 공개 자료이므로 판단에서 제외",
    "observed_after_cutoff": "질문 기준 이후 관측 자료이므로 판단에서 제외",
    "publication_time_unknown": "게시 시각을 확인할 수 없어 판단에서 제외",
    "publication_date_unverified": "게시일의 출처를 검증하지 못해 판단에서 제외",
    "publication_timezone_unknown": "게시일의 시간대가 미상이라 판단에서 제외",
    "publication_time_unknown_at_cutoff": "게시 날짜만 있어 기준 시각 이전 공개 여부가 미상인 자료 제외",
}


def _aware_datetime(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("Evidence cutoff and timestamps must include timezone")
    return parsed


def _publication_reason(source, cutoff):
    value = source.get("published_at")
    if not value:
        return "publication_time_unknown"
    if source.get("date_verified") is False:
        return "publication_date_unverified"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{8}", value):
        if not source.get("publication_timezone"):
            return "publication_timezone_unknown"
        # Day precision gives a range, never a fabricated midnight publication.
        zone = ZoneInfo(source["publication_timezone"])
        day = date.fromisoformat(value)
        start = datetime.combine(day, time.min, zone)
        end = datetime.combine(day + timedelta(days=1), time.min, zone)
        if cutoff < start:
            return "published_after_cutoff"
        if cutoff < end:
            return "publication_time_unknown_at_cutoff"
        return None
    return "published_after_cutoff" if _aware_datetime(value) > cutoff else None


def filter_evidence(evidence, as_of):
    """Return copies of eligible rows and only whitelisted metadata for exclusions.

    Day-only publication dates require a source timezone and the entire source
    day to precede the cutoff. This checks recorded times, not historical page
    versions, statistical vintages, relevance, or factual correctness.
    """
    cutoff = _aware_datetime(as_of)
    visible, excluded = [], []
    for raw in evidence:
        source = deepcopy(raw)
        reason = _publication_reason(source, cutoff)
        value = source.get("data_as_of")
        if value:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                observed_day = date.fromisoformat(value)
                future = observed_day > cutoff.date()
            else:
                observed = _aware_datetime(value)
                observed_day = observed.astimezone(cutoff.tzinfo).date()
                future = observed > cutoff
            if future and reason is None:
                reason = "observed_after_cutoff"
            source["observation_scope"] = "prior_date" if observed_day < cutoff.date() else "cutoff_date"
        if reason:
            excluded.append({"id": raw["id"], "published_at": raw.get("published_at"),
                             **({"data_as_of": raw["data_as_of"]} if "data_as_of" in raw else {}),
                             **({"publication_timezone": raw["publication_timezone"]}
                                if "publication_timezone" in raw else {}),
                             "reason": reason})
        else:
            visible.append(source)
    return visible, excluded
