"""Small, source-bound claim vocabulary; rules label excerpts, not verified truth."""
import hashlib
import json
import math
import re
from decimal import Decimal
from html import unescape
from typing import Literal

from pydantic import BaseModel, ConfigDict

NEGATION_CUES = (r"\b(?:not(?! only\b)|no|never|denied|unconfirmed|unverified|cancelled|canceled)\b|"
                 r"않|아니|없|미확인|미정|부인|취소")
TENTATIVE_CUES = (r"\b(?:if|unless|may|might|could|would|will|expects?|expected|forecast\w*|"
                  r"project\w*|estimat\w*|plans?|planned|guidance|targets?|aims?)\b|"
                  r"예정|계획|전망|예상|추정|목표|조건|경우|가능|검토")
APPROVAL_ELLIPSIS = r"\b(?:maybe|perhaps)\s+([a-z][a-z0-9-]*)\s+as well\b"


class EvidenceClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    source_id: str
    source_sha256: str
    span_start: int
    span_end: int
    subject: str
    predicate: Literal["share_gift", "growth_outlook", "exchange_rate", "document_context",
                       "bibliography", "technical_description", "production_announcement",
                       "earnings_schedule", "financial_results", "economic_report", "policy_decision",
                       "economic_metric", "stock_price", "regulatory_approval", "investment_commentary", "analyst_opinion"]
    kind: Literal["event", "forecast", "observation", "context", "technical", "financial", "price", "schedule", "opinion"]
    epistemic_status: Literal["reported", "possible", "not_applicable"]
    value: float | None
    unit: str | None
    time_text: str | None
    published_at: str | None
    method: Literal["excerpt_rules_v1", "excerpt_rules_v2", "excerpt_rules_v3", "excerpt_rules_v4"] = "excerpt_rules_v3"


def readable_excerpt(quote):
    # Display-only projection; the stored citation and offsets remain byte-for-byte text slices.
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]*>", "", quote))).strip(" .\n□")


def sentence_spans(text):
    """Keep exact offsets and abbreviations shared by quote windows and claim guards."""
    start = 0
    # Flattened newsroom bullets can begin a new By-ing clause without a period.
    for match in re.finditer(r"[.!?;](?=\s|$)\s*|\s+(?=By [a-z]+ing\b)", text):
        end = match.end()
        if re.search(r"\b(?:Inc|Corp|Ltd|Apr|Jan|Feb|Mar|Jun|Jul|Aug|Sep|Oct|Nov|Dec|e\.g|i\.e|a\.m|p\.m)\.\s*$",
                     text[start:end], re.I):
            continue
        yield start, end
        start = end
    if start < len(text):
        yield start, len(text)


def claim_key(claim):
    """Keep approval/rating subjects and states distinct; preserve older topic keys."""
    if claim is None:
        return None
    if claim.predicate in {"regulatory_approval", "analyst_opinion"}:
        return f"{claim.predicate}:{claim.subject.casefold()}:{claim.epistemic_status}"
    return claim.predicate


def approval_subject(text):
    # ponytail: narrow passive product/PDUFA patterns; broader regulatory language stays unknown.
    matches = list(re.finditer(
        r"\b([a-z][a-z0-9-]*)\s+(?:(?:(?:is|was|has been|may be|might be|could be|will be)\s+)?approved\b|"
        r"(?:is\s+)?(?:headed for|expected to receive)\s+(?:a\s+)?(?:likely\s+|possible\s+)?approval\b)", text, re.I))
    if len(matches) != 1:
        return None
    subject = matches[0].group(1)
    if not (re.search(r"\bpdufa\b", text, re.I) or re.search(
            r"\b(?:drug|therapy|treatment|vaccine)\s+" + re.escape(subject) + r"\b", text, re.I)):
        return None
    return subject.casefold()


def claim_for_quote(source, quote):
    """Only recognize supported, narrow excerpt patterns; unknown stays unknown.

    ponytail: Narrow excerpt patterns cover the pilot without a graph database.
    Unrecognized or mixed excerpts need semantic review, not a guessed graph edge.
    """
    if not quote.strip() or quote not in source["text"]:
        return None
    text = readable_excerpt(quote)
    start = source["text"].index(quote)
    context = " ".join(readable_excerpt(source["text"][a:b]) for a, b in sentence_spans(source["text"])
                       if a < start + len(quote) and b > start)
    metadata = source.get("metadata", {})
    subject = predicate = kind = status = None
    value = unit = time_text = None
    if source["kind"] == "metric":
        try:
            data = json.loads(source["text"])
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        # Bind only fields actually contained in the selected exact excerpt.
        if (isinstance(data.get("unit"), str) and data["unit"] and type(data.get("current_value")) in (int, float)
                and math.isfinite(data["current_value"]) and isinstance(data.get("as_of"), str)
                and all(json.dumps(data.get(k), ensure_ascii=False) in quote
                        for k in ("current_value", "unit", "as_of")) and data.get("as_of")):
            fx = data["unit"] == "KRW/USD"
            subject = "원/달러 환율" if fx else source["id"].rsplit(":", 1)[-1]
            predicate, kind, status = "exchange_rate" if fx else "economic_metric", "observation", "reported"
            value, unit, time_text = data["current_value"], data["unit"], data["as_of"]
    elif source["kind"] == "document":
        product = approval_subject(text)
        if product and not re.search(NEGATION_CUES, context, re.I):
            continuation = re.search(APPROVAL_ELLIPSIS, text, re.I)
            subject = continuation.group(1).casefold() if continuation else product
            predicate, kind = "regulatory_approval", "event"
            status = "possible" if re.search(TENTATIVE_CUES + r"|\blikely\b|\bmaybe\b|\bperhaps\b|\bheaded for\b|\?", context, re.I) else "reported"
        elif re.search(r"\$([a-z][a-z0-9.]*)\s+is rated (?:buy|sell|hold)\b", text, re.I) and re.search(r"\banalysts?\b", text, re.I):
            subject = re.search(r"\$([a-z][a-z0-9.]*)\s+is rated", text, re.I).group(1).upper()
            predicate, kind, status = "analyst_opinion", "opinion", "reported"
        elif re.search(r"\b(?:stock forum|defensive dividend growth stocks|current price\s*\$|shares\s*\$)\b", text, re.I):
            subject, predicate, kind, status = "투자 관련 게시글", "investment_commentary", "opinion", "reported"
        elif re.search(r"^(?:Note: This report was prepared|This document summarizes comments|Overall Economic Activity$)", text):
            subject, predicate, kind, status = "문서 작성 안내", "document_context", "context", "not_applicable"
        elif re.search(r"증여(?:받|한|하|했|할|해|되)", text) and re.search(r"주식|지분", text) and not re.search(r"종가|주가|거래가격", text):
            subject_match = re.search(r"([\w·]+)\s+(?:주식|지분)", text)
            subject = subject_match.group(0) if subject_match else "주식·지분"
            predicate, kind = "share_gift", "event"
            status = "possible" if re.search(r"가능|검토|예정|계획", text) else "reported"
        elif "성장률" in text and re.search(r"전망|예상|예측", text) and not re.search(r"실제|실현|달성|기록했", text):
            subject, predicate, kind, status = "경제 성장률", "growth_outlook", "forecast", "possible"
            period = re.search(r"(?:금년|올해)(?:\s*및\s*내년)?|내년|\d{4}년", text)
            time_text = period.group(0) if period else None
        elif "환율" in text and re.search(r"하락|상승|절하|절상", text):
            # Mixed policy/FX excerpts are left unclassified instead of imposing one type.
            if not re.search(r"성장률|통화정책|기준금리", text):
                subject = "원/달러 환율" if re.search(r"원\s*/\s*달러", text) else "환율"
                predicate = "exchange_rate"
                if re.search(r"전망|예상|예측|가능", text):
                    kind, status = "forecast", "possible"
                elif re.search(r"하였다|했다|하였|되었다|됐다|기록", text):
                    kind, status = "observation", "reported"
        if not predicate:
            paper = metadata.get("source_type") in ("paper", "standard", "jedec")
            references = list(re.finditer(r"(?:^|\n)[ \t]*\[\d+\]", source["text"]))
            reference_start = references[0].start() if len(references) >= 3 else len(source["text"])
            prelude = source["text"][:reference_start]
            if reference_start < 240 and re.search(r"ACM|IEEE|doi\.org", prelude) and re.search(r"\b(?:19|20)\d{2}\b", prelude):
                reference_start = 0  # A clipped opening citation before the first numbered entry.
            section = re.search(r"(?im)^\s*(?:appendix|appendices|supplementary|conclusions?|discussion|부록|결론)\b",
                                source["text"][reference_start:])
            reference_end = reference_start + section.start() if section else len(source["text"])
            if paper and len(references) >= 3 and reference_start <= start < reference_end:
                subject, predicate, kind, status = "참고문헌 목록", "bibliography", "context", "not_applicable"
            elif re.search(r"(?:revenues?|operating profit|net profit|매출|영업이익|순이익).*?\d", text, re.I):
                subject, predicate, kind, status = "기업 실적", "financial_results", "financial", "reported"
            elif re.search(r"(?:begun|begins?|started) mass production|양산(?:을)?\s*(?:시작|개시)", text, re.I):
                subject, predicate, kind, status = "양산 발표", "production_announcement", "event", "reported"
            elif re.search(r"(?:will present.*?earnings|conference call will be held|실적.*?발표.*?예정)", text, re.I):
                subject, predicate, kind, status = "실적 발표 일정", "earnings_schedule", "schedule", "reported"
            elif paper and re.search(r"bandwidth|memory|interconnect|DRAM|HBM|메모리|대역폭", text, re.I):
                subject, predicate, kind, status = "기술 문헌", "technical_description", "technical", "reported"
            elif re.search(r"기준금리.*?(?:상향|하향|유지).*?하기로", text):
                subject, predicate, kind, status = "기준금리 결정", "policy_decision", "event", "reported"
            elif re.search(r"health care and insurance cost pressures|business activity (?:rose|increased)", text, re.I):
                subject, predicate, kind, status = "경제 현황", "economic_report", "observation", "reported"
            elif (re.search(r"\d{4}[-./]\d{2}[-./]\d{2}", text)
                    and re.search(r"(?:(?:종가|주가)(?:는|은|가|이)?\s*|(?:share price|closing price)(?: was| is| closed at| of|:)?\s+)"
                                  r"\d[\d,.]*\s*(?:원|달러|USD|KRW)", text, re.I)
                    and re.search(r"마감(?:했다|하였다|됐|되었)|거래됐다|closed", text, re.I)):
                subject, predicate, kind, status = "주식 거래가격", "stock_price", "price", "reported"
    if not all((subject, predicate, kind, status)):
        return None
    # ponytail: conservative sentence-level guards; mixed/unsupported assertions
    # stay extractive/unknown instead of guessing a negated or conditional graph edge.
    negated = re.search(NEGATION_CUES, context, re.I)
    tentative = re.search(TENTATIVE_CUES, context, re.I)
    if predicate in {"production_announcement", "financial_results", "stock_price", "policy_decision", "economic_report"} and (negated or tentative):
        return None
    if predicate == "earnings_schedule" and negated:
        return None
    if predicate == "analyst_opinion" and negated:
        return None
    digest = hashlib.sha256(source["text"].encode("utf-8")).hexdigest()
    identity = json.dumps([source["id"], metadata.get("url"), digest, start, len(quote), predicate], ensure_ascii=False)
    return EvidenceClaim(claim_id="claim:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        source_id=source["id"], source_sha256=digest, span_start=start, span_end=start + len(quote),
        subject=subject, predicate=predicate, kind=kind, epistemic_status=status,
        value=value, unit=unit, time_text=time_text,
        published_at=metadata.get("published_at") or metadata.get("published_date"),
        method="excerpt_rules_v4" if predicate in {"regulatory_approval", "investment_commentary", "analyst_opinion"} else "excerpt_rules_v3")


def claim_policy(claim):
    if claim is None:
        return None
    material = {"event": "event_report", "forecast": "economic_forecast",
                "observation": "economic_observation", "context": "other", "technical": "technical_document",
                "financial": "financial_report", "price": "price_observation", "schedule": "event_report",
                "opinion": claim.predicate}[claim.kind]
    if claim.predicate == "bibliography":
        material = "reference_list"
    return material, claim.epistemic_status


def claim_explanation(claim, quote):
    """A readable projection of one supported claim, not an LLM paraphrase."""
    if claim.value is not None:
        return f"{claim.time_text} 기준 {claim.subject}은 {claim.value} {claim.unit}로 제공됐다. 해당 시점의 지표값이다."
    excerpt = readable_excerpt(quote)
    if claim.predicate == "regulatory_approval":
        limit = ("승인 가능성에 관한 추측이며 승인 완료를 뜻하지 않는다." if claim.epistemic_status == "possible" else
                 "출처가 전한 승인 보도이며 공식 확인이나 현재 승인 상태를 뜻하지 않는다.")
        return f"{claim.subject} 승인 관련 원문: {excerpt}. {limit} 제품의 회사 소속은 이 주장으로 확정하지 않는다."
    labels = {
        "analyst_opinion": ("분석가 의견 원문", "평가·목표에 관한 의견이며 실현 수익률이나 기업 실적 관측값이 아니다. 전망 기간은 별도 확인이 필요하다."),
        "investment_commentary": ("투자 해설·게시글 원문", "투자 관련 언급이며 실제 거래가격·기업 실적·거시경제 관측값을 확인한 것이 아니다."),
        "bibliography": ("참고문헌 목록", "다른 문헌의 서지정보이며 이 논문의 실험 성과를 확인하는 구간은 아니다."),
        "technical_description": ("기술 문헌 원문", "연구·설계 설명이며 제품 승인이나 주가 관측을 확인하는 구간은 아니다."),
        "production_announcement": ("양산 발표 원문", "양산 발표이며 고객 인증이나 주가 영향을 확정하지 않는다."),
        "earnings_schedule": ("실적 발표 일정 원문", "발표 예정 안내이며 실적 수치가 발표됐다는 뜻은 아니다."),
        "financial_results": ("기업 실적 원문", "실적 수치이며 주식 거래가격이 아니다."),
        "economic_report": ("경제 현황 원문", "출처의 보고 내용이며 특정 기업의 실적 영향은 이 구간으로 확정하지 않는다."),
        "policy_decision": ("정책 결정 원문", "이 구간에 없는 환율·산업·기업 영향을 덧붙이지 않는다."),
        "stock_price": ("주식 가격 원문", "해당 시점의 가격 보도이며 향후 수익을 보장하지 않는다."),
    }
    if claim.predicate in labels:
        prefix, limit = labels[claim.predicate]
        return f"{prefix}: {excerpt}. {limit}"
    prefix = {"event": "증여 관련 원문 보도", "forecast": "경제 전망 원문",
              "observation": "환율 관측 원문 설명", "context": "문서 작성 안내 원문"}[claim.kind]
    limit = {"event": "보도 내용이며 실제 이행 여부나 주가 영향을 확정하지 않는다.",
             "forecast": "전망치이며 실현된 관측값이 아니다.",
             "observation": "이 구간에 없는 정책·기업 영향은 덧붙이지 않는다.",
             "context": "경기 방향을 관측한 주장이 아니다."}[claim.kind]
    return f"{prefix}: {excerpt}. {limit}"


def claim_issues(source, review):
    """Rebuild trusted fields, then screen limited interpretation contradictions.

    These checks are not a general entailment judge. Free prose still needs review.
    """
    expected = claim_for_quote(source, review.quote)
    issues = []
    if review.claim != expected:
        issues.append("claim_binding_mismatch")
    if expected is None:
        return issues
    if (review.material_type, review.claim_status) != claim_policy(expected):
        issues.append("claim_type_status_mismatch")
    explanation = review.interpretation
    if explanation != claim_explanation(expected, review.quote):
        issues.append("claim_explanation_mismatch")
    if expected.predicate == "share_gift" and "증여" not in explanation:
        issues.append("claim_subject_missing")
    if expected.predicate == "growth_outlook" and not (
            "성장" in explanation and re.search(r"전망|예상|예측", explanation)):
        issues.append("claim_subject_missing")
    if expected.predicate == "exchange_rate" and not re.search(r"환율|KRW/USD", explanation):
        issues.append("claim_subject_missing")
    if expected.kind == "forecast" and re.search(r"달성했다|달성하였다|실현됐다|확정됐다|기록했다", explanation):
        issues.append("forecast_presented_as_realized")
    if expected.kind != "context":
        numbers = lambda text: {Decimal(n.replace(",", "")) for n in
            re.findall(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?", text)}
        # Restrict to the selected excerpt, including dates; publication != observation.
        if numbers(explanation) - numbers(review.quote):
            issues.append("interpretation_number_outside_quote")
        for topic in ("통화정책", "기준금리", "성장률", "물가상승률", "가계대출", "주택가격"):
            if topic in explanation and topic not in review.quote:
                issues.append("interpretation_topic_outside_quote:" + topic)
    return issues
