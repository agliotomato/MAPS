"""Fixed-input contribution adapter. CLI is offline; no retrieval or DB writes.

Availability metadata is checked for consistency, not independently certified.
Historical cohort/vintage review and production-report equivalence remain gates.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'semiconductor_agent_design'))
from app.forecast.price_forecast import _compute_price_features, forecast_from_inputs, select_price_history
from role_prompts import BOUNDARY

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

from independent_reasoning import (Analysis, ROLES, analyst_prompt, company_attribution_review,
                                   limitation_issues, price_window, price_history_issues, price_claim_denied, PRICE_TREND_RULES,
                                   counter_condition_issues, COUNTER_CONDITION_RULES, price_threshold_issues, PRICE_THRESHOLD_RULES,
                                   economy_support_issues, price_only_forecast_issues, MARKET_FORWARD_RULES)
from financial_claims import financial_facts, FINANCIAL_SCOPE_RULES

PROTOCOL = ROOT / 'benchmarks/contribution_protocol_v1.json'
HORIZONS = (7, 14, 30)
KST = timezone(timedelta(hours=9))
SOURCE_FIELDS = {'id', 'role', 'kind', 'origin', 'parents', 'payload', 'available_at',
                 'precision', 'source_ref', 'version_ref', 'availability_basis'}
KINDS = {'news', 'disclosure', 'economy', 'technology', 'price', 'derived', 'event_outcome', 'engine_signal'}
FINAL_RULES = (BOUNDARY + '제공 기준일의 직전 종가 대비 7·14·30달력일 후 방향을 판단한다. '
    '목표일이 휴장일이면 이후 첫 거래일 종가다. UP은 상승, NON_UP은 비상승, ABSTAIN은 판단 유보다. '
    'observation에는 확인된 관측과 ID, hypothesis에는 조건부 전망, counter_condition에는 반대 조건, '
    'limitations에는 실제 확인하지 못한 항목을 구체적인 문장으로 쓴다. null, /, 없음은 금지한다. '
    '자료 부재와 분석 미검증을 구분한다. 근거 부재를 횡보로 해석하지 않는다. '
    'price_windows의 날짜·관측 건수·날짜 간격을 사용하고 향후 전망 기간을 과거 관측 기간으로 쓰지 않는다. '
    'price_features의 기간 단위는 price_feature_units를 따른다. '
    '독립 분석은 검증되지 않은 가설이다. 단순 다수결 대신 원문·시점·중복·기간을 확인한다. '
    'source_ids는 실제 사용한 원자료 ID, selected_hypotheses는 채택한 role:d7 같은 ID다. '
    '독립 분석이 없거나 직접 추론하면 selected_hypotheses를 비운다. JSON만 반환한다.')
FINAL_RULES += FINANCIAL_SCOPE_RULES + PRICE_TREND_RULES + COUNTER_CONDITION_RULES + PRICE_THRESHOLD_RULES + (' 타사 실적은 대상 회사 실적이 아니다. 실적 문장의 회사명을 명시한다. '
    '타사 근거의 hypothesis에는 원자료 회사·대상 회사와 조건부 전달 경로를 모두 적는다. '
    'excluded_hypotheses에 있는 분석은 최종 판단 근거로 채택하지 않는다.')


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('Timestamp must be explicit, not inferred')
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError('Timezone required')
    return dt


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_packet(packet, *, development=False):
    """Reject inconsistent/unbounded inputs; never elevate declarations to certification."""
    if set(packet) != {'id', 'company_code', 'as_of', 'evidence', 'limitations'}:
        raise ValueError('Unexpected packet fields; gold and audit findings belong outside inputs')
    if not packet['id'] or packet['company_code'] not in ('005930', '000660', '042700'):
        raise ValueError('Unknown case/target')
    cutoff = timestamp(packet['as_of'])
    if cutoff.astimezone(KST).time().isoformat() != '09:00:00':
        raise ValueError('This protocol requires 09:00 KST')
    if (not isinstance(packet['limitations'], list) or not all(isinstance(x, str) for x in packet['limitations'])
            or not set(packet['limitations']) <= {'SYNTHETIC_ONLY', 'VINTAGE_UNVERIFIED',
                'HUMAN_REVIEW_PENDING', 'CALENDAR_UNVERIFIED', 'NOT_EVALUATION_ELIGIBLE',
                'PRICE_AVAILABILITY_UNVERIFIED', 'PRICE_ADJUSTMENT_UNVERIFIED'}):
        raise ValueError('Use neutral limitation codes; factual summaries need source lineage')
    reconstruction = development and {'NOT_EVALUATION_ELIGIBLE', 'VINTAGE_UNVERIFIED',
        'PRICE_AVAILABILITY_UNVERIFIED', 'PRICE_ADJUSTMENT_UNVERIFIED'} <= set(packet['limitations'])
    sources = packet['evidence']
    if not isinstance(sources, list) or not sources:
        raise ValueError('Missing evidence')
    by_id, availability = {}, {}
    for s in sources:
        if (set(s) != SOURCE_FIELDS or not isinstance(s['id'], str) or not s['id']
                or s['id'] in by_id or s['role'] not in ROLES or s['kind'] not in KINDS):
            raise ValueError('Invalid/duplicate source')
        if any(not isinstance(s[k], str) or not s[k].strip()
               for k in ('origin', 'source_ref', 'version_ref', 'availability_basis')):
            raise ValueError('Missing availability/version/origin record')
        if not isinstance(s['parents'], list) or not all(isinstance(x, str) for x in s['parents']):
            raise ValueError('Invalid lineage')
        if len(set(s['parents'])) != len(s['parents']):
            raise ValueError('Duplicate lineage edge')
        if s['kind'] in ('derived', 'engine_signal', 'event_outcome') and not s['parents']:
            raise ValueError('Derived evidence needs complete parent lineage')
        unknown_price_time = reconstruction and s['kind'] == 'price' and s['available_at'] is None and s['precision'] == 'unknown'
        available = None if unknown_price_time else timestamp(s['available_at'])
        if unknown_price_time:
            pass  # Preserve unknown receipt time; never infer it from the trade date.
        elif s['precision'] == 'date':
            # A date denotes an uncertain time within that LOCAL day, not its start.
            if available.time().isoformat() != '00:00:00':
                raise ValueError('Date precision must record local midnight and timezone')
            available += timedelta(days=1)
        elif s['precision'] != 'timestamp':
            raise ValueError('Unknown publication precision')
        if available is not None and available >= cutoff:
            raise ValueError('Evidence is not strictly available before cutoff')
        if s['kind'] == 'event_outcome':
            for key in ('window_end_available_at', 'computed_at'):
                if timestamp(s['payload'][key]) >= cutoff or timestamp(s['payload'][key]) > available:
                    raise ValueError('Outcome window/computation crosses cutoff')
        by_id[s['id']] = s
        availability[s['id']] = available
    done, visiting = set(), set()

    def visit(source_id):
        if source_id not in by_id or source_id in visiting:
            raise ValueError('Missing parent or cyclic lineage')
        if source_id in done:
            return
        visiting.add(source_id)
        for parent in by_id[source_id]['parents']:
            if parent not in by_id:
                raise ValueError('Missing parent')
            if availability[parent] is None or availability[source_id] is None or availability[parent] > availability[source_id]:
                raise ValueError('Derived source predates its parent')
            visit(parent)
        visiting.remove(source_id)
        done.add(source_id)

    for source_id in by_id:
        visit(source_id)
    prices = [s for s in sources if s['kind'] == 'price']
    if len(prices) != 1 or prices[0]['role'] != 'market' or prices[0]['parents']:
        raise ValueError('Exactly one original market price series required')
    p = prices[0]['payload']
    if set(p) != {'company_code', 'rows', 'adjustment', 'calendar_ref', 'expected_sessions', 'corporate_action_review'}:
        raise ValueError('Missing price audit fields')
    if p['company_code'] != packet['company_code'] or any(not p[k] for k in ('adjustment', 'calendar_ref', 'corporate_action_review')):
        raise ValueError('Price target or provenance missing')
    dates = []
    for row in p['rows']:
        if set(row) != {'trade_date', 'close', 'volume', 'available_at'}:
            raise ValueError('Unexpected price fields')
        trade_day = date.fromisoformat(row['trade_date'])
        unknown_row_time = reconstruction and prices[0]['available_at'] is None and row['available_at'] is None
        row_available = None if unknown_row_time else timestamp(row['available_at'])
        close_time = datetime.combine(trade_day, datetime.min.time(), KST) + timedelta(hours=15, minutes=30)
        if (trade_day >= cutoff.astimezone(KST).date()
                or (not unknown_row_time and (row_available < close_time or row_available >= cutoff
                    or row_available > timestamp(prices[0]['available_at'])))
                or not finite(row['close']) or row['close'] <= 0
                or not finite(row['volume']) or row['volume'] < 0):
            raise ValueError('Invalid or future price row')
        dates.append(row['trade_date'])
    if not dates or dates != sorted(set(dates)) or dates != p['expected_sessions']:
        raise ValueError('Unsorted/duplicate/missing price sessions')
    signals = [s for s in sources if s['kind'] == 'engine_signal']
    components = []
    for s in signals:
        component, values = s['payload']['component'], s['payload']['values']
        expected = {'events': ('news', 'event_bias', 1), 'tech': ('semiconductor', 'tech_bias', 1),
                    'market': ('economy', 'macro_bias', .15)}
        if component not in expected or not isinstance(values, dict):
            raise ValueError('Unknown engine signal')
        role, key, bound = expected[component]
        if s['role'] != role or not finite(values.get(key)) or abs(values[key]) > bound:
            raise ValueError('Invalid engine signal value')
        components.append(component)
    if len(set(components)) != len(components):
        raise ValueError('Duplicate engine components')
    json.dumps(packet, allow_nan=False)  # Reject nonfinite values anywhere before a callback.
    return by_id


def without_news(packet):
    """Remove news, aliases by origin/exact payload, and all transitive derivatives."""
    result = deepcopy(packet)
    sources = result['evidence']
    removed = {s['id'] for s in sources if s['kind'] == 'news'}
    while True:
        origins = {s['origin'] for s in sources if s['id'] in removed}
        payloads = {sha(s['payload']) for s in sources if s['id'] in removed}
        expanded = removed | {s['id'] for s in sources if s['origin'] in origins
                             or sha(s['payload']) in payloads or set(s['parents']) & removed}
        if expanded == removed:
            break
        removed = expanded
    result['evidence'] = [s for s in sources if s['id'] not in removed]
    # No excluded titles, snippets or IDs are handed back to the model.
    return result, sorted(removed)


def history(packet, horizon):
    rows = next(s['payload']['rows'] for s in packet['evidence'] if s['kind'] == 'price')
    return select_price_history(rows, timestamp(packet['as_of']).astimezone(KST), max(90, horizon * 4))


def model_packet(packet, role=None):
    value = deepcopy(packet)
    if role:
        value['evidence'] = [s for s in value['evidence'] if s['role'] == role]
    value['financial_facts'] = financial_facts(value['evidence'])
    value['price_features'] = ({str(h): _compute_price_features(history(packet, h)) for h in HORIZONS}
                               if any(s['kind'] == 'price' for s in value['evidence']) else {})
    value['price_windows'] = [w for s in value['evidence'] if (w := price_window(s))]
    value['forward_support_by_horizon'] = {f'd{h}': dict(
        issues=price_only_forecast_issues(value['evidence']), policy=MARKET_FORWARD_RULES)
        for h in HORIZONS}
    if role == 'economy':
        value['economic_support_by_horizon'] = {f'd{h}': dict(issues=economy_support_issues(
            packet['company_code'], value['evidence'], h, packet['as_of'])) for h in HORIZONS}
    if value['price_features']:
        value['price_feature_units'] = ('return_7d/20d/60d use 7/20/60 observed row intervals, not calendar days. '
            'An unavailable return window is null, not an observed flat return. The price engine requires '
            'observed 7/20 returns and renormalizes its existing momentum weights over available returns. '
            'Other rolling features use up to the stated number of rows. Features for horizon h first restrict '
            'history to max(90, h*4) calendar days before as_of, extending to the latest 61 eligible prior '
            'observations when needed and available. Fewer observations stay missing, never filled. '
            'price_windows describes the full delivered rows; 60 observed intervals can span more calendar days. '
            'Horizon keys 7/14/30 are future calendar days.')
    return value


class FinalRow(BaseModel):
    model_config = ConfigDict(extra='forbid')
    direction: Literal['UP', 'NON_UP', 'ABSTAIN']
    source_ids: list[str]
    selected_hypotheses: list[str]
    observation: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    counter_condition: str = Field(min_length=1)
    limitations: str = Field(min_length=1)


class FinalReport(BaseModel):
    model_config = ConfigDict(extra='forbid')
    predictions: dict[Literal['7', '14', '30'], FinalRow]

    @classmethod
    def model_json_schema(cls, **kwargs):
        schema = super().model_json_schema(**kwargs)
        schema['properties']['predictions'] = dict(type='object',
            properties={str(h): {'$ref': '#/$defs/FinalRow'} for h in HORIZONS},
            required=[str(h) for h in HORIZONS], additionalProperties=False)
        return schema


def validate_analysis(raw, source_ids):
    value = Analysis.model_validate_json(raw).model_dump()
    for row in value.values():
        ids = ([row['primary_evidence_id']] if row['primary_evidence_id'] else []) + row['supporting_evidence_ids']
        if not set(ids) <= source_ids or any(not row[k].strip() for k in ('mechanism', 'counter_condition', 'missing')):
            raise ValueError('Invalid analysis evidence/text')
        if row['primary_evidence_id'] is None and (ids or row['direction'] != '판단 유보'):
            raise ValueError('Missing primary evidence requires abstention')
    return value


def validate_final(raw, source_ids, notes):
    report = FinalReport.model_validate_json(raw).model_dump()
    if set(report['predictions']) != {str(h) for h in HORIZONS}:
        raise ValueError('Missing horizon')
    candidates = {f"{n['role']}:{h}": row for n in notes for h, row in n['analysis'].items()}
    for horizon, row in report['predictions'].items():
        ids, selected = row['source_ids'], row['selected_hypotheses']
        if (len(set(ids)) != len(ids) or not set(ids) <= source_ids
                or len(set(selected)) != len(selected)
                or any(not row[k].strip() for k in ('observation', 'hypothesis', 'counter_condition', 'limitations'))
                or (row['direction'] != 'ABSTAIN' and not ids)):
            raise ValueError('Invalid final evidence/text')
        for key in selected:
            note = candidates.get(key)
            if (note is None or not key.endswith(f':d{horizon}') or not note['primary_evidence_id']
                    or note['primary_evidence_id'] not in ids):
                raise ValueError('Unbound selected hypothesis')
            direction = {'Bullish': 'UP', 'Neutral': 'NON_UP', 'Bearish': 'NON_UP'}.get(note['direction'])
            if direction != row['direction']:
                raise ValueError('Selected hypothesis direction differs from final')
    return report


def final_schema(source_ids, notes, excluded=()):
    """Offer actual per-horizon hypothesis IDs instead of unconstrained examples."""
    schema = FinalReport.model_json_schema()
    directions = {'Bullish':'UP', 'Neutral':'NON_UP', 'Bearish':'NON_UP'}
    for h in HORIZONS:
        variants = []
        for direction in ('UP', 'NON_UP', 'ABSTAIN'):
            row = deepcopy(schema['$defs']['FinalRow'])
            row['properties']['direction']['enum'] = [direction]
            row['properties']['source_ids']['items'] = dict(type='string', enum=sorted(source_ids))
            if direction != 'ABSTAIN':
                row['properties']['source_ids']['minItems'] = 1
            eligible = [f"{n['role']}:d{h}" for n in notes
                        if f"{n['role']}:d{h}" not in excluded and n['analysis'][f'd{h}']['primary_evidence_id'] and
                        directions.get(n['analysis'][f'd{h}']['direction']) == direction]
            selected = row['properties']['selected_hypotheses']
            if eligible:
                selected['items'] = dict(type='string', enum=eligible)
            else:
                selected['maxItems'] = 0
            variants.append(row)
        schema['properties']['predictions']['properties'][str(h)] = dict(anyOf=variants)
    return schema


def analysis_attribution_checks(packet, notes):
    # Keep the persisted review key; also check the shared limits and price window.
    checks = {}
    for note in notes:
        for horizon, row in note['analysis'].items():
            if row['direction'] == '판단 유보':
                continue
            ids = {row['primary_evidence_id'], *row['supporting_evidence_ids']}
            refs = [s for s in packet['evidence'] if s['id'] in ids]
            check = company_attribution_review(packet['company_code'],
                [s for s in packet['evidence'] if s['id'] in ids], row['mechanism'])
            check['issues'].extend(limitation_issues(row['missing']))
            check['issues'].extend(price_only_forecast_issues(refs))
            check['issues'].extend(counter_condition_issues(row['direction'], row['mechanism'], row['counter_condition']))
            if note['role'] == 'economy':
                check['issues'].extend(economy_support_issues(packet['company_code'],
                    [s for s in packet['evidence'] if s['id'] in ids], int(horizon[1:]), packet.get('as_of')))
            if note['role'] == 'market':
                check['issues'].extend(price_threshold_issues(row['mechanism'], row['counter_condition'], row['missing']))
            if note['role'] == 'market' and row['direction'] == 'Neutral' and price_claim_denied(row['mechanism']):
                check['issues'].append('uncertainty_is_not_neutral')
            for source in packet['evidence']:
                if source['id'] == row['primary_evidence_id']:
                    check['issues'].extend(price_history_issues(row['mechanism'], source))
            if check['issues']:
                check['status'] = 'needs_review'
            checks[f"{note['role']}:{horizon}"] = check
    return checks


def final_attribution_checks(report, packet, notes):
    analyses = analysis_attribution_checks(packet, notes)
    checks = {}
    for horizon, row in report['predictions'].items():
        check = company_attribution_review(packet['company_code'],
            [s for s in packet['evidence'] if s['id'] in row['source_ids']], row['hypothesis'], row['observation'])
        check['issues'].extend(limitation_issues(row['limitations']))
        check['issues'].extend(counter_condition_issues(row['direction'], row['hypothesis'], row['counter_condition']))
        check['issues'].extend(price_threshold_issues(row['hypothesis'], row['counter_condition'], row['limitations']))
        selected_sources = [s for s in packet['evidence'] if s['id'] in row['source_ids']]
        if row['direction'] != 'ABSTAIN':
            check['issues'].extend(price_only_forecast_issues(selected_sources))
        documents = [s for s in selected_sources if s['kind'] != 'price']
        if any(key.startswith('economy:') for key in row['selected_hypotheses']) or (
                documents and all(s.get('role') == 'economy' for s in documents)):
            check['issues'].extend(economy_support_issues(packet['company_code'], selected_sources,
                int(horizon), packet.get('as_of')))
        for source in packet['evidence']:
            if source['id'] in row['source_ids']:
                check['issues'].extend(price_history_issues(row['observation'], source))
                check['issues'].extend(price_history_issues(row['hypothesis'], source))
        for selected in row['selected_hypotheses']:
            check['issues'].extend(f"{selected}:{issue}" for issue in analyses.get(selected, {}).get('issues', []))
        if check['issues']:
            check['status'] = 'needs_review'
        checks[horizon] = check
    return checks


def bind_final(raw, source_ids, notes, packet=None):
    """Compile citations already implied by selected hypotheses, preserving prose.

    Raw provider bytes remain in raw_final. This resolves redundant references;
    it does not verify the hypothesis's facts, target relevance or reasoning.
    """
    value = FinalReport.model_validate_json(raw).model_dump()
    candidates = {f"{n['role']}:{h}":row for n in notes for h,row in n['analysis'].items()}
    adjustments = []
    for horizon, row in value['predictions'].items():
        before = dict(source_ids=list(row['source_ids']), selected_hypotheses=list(row['selected_hypotheses']))
        row['source_ids'] = list(dict.fromkeys(row['source_ids']))
        row['selected_hypotheses'] = list(dict.fromkeys(row['selected_hypotheses']))
        for key in row['selected_hypotheses']:
            note = candidates.get(key)
            direction = None if note is None else {'Bullish':'UP','Neutral':'NON_UP','Bearish':'NON_UP'}.get(note['direction'])
            if (note is None or not key.endswith(f':d{horizon}') or direction != row['direction']
                    or note['primary_evidence_id'] not in source_ids):
                raise ValueError('Cannot bind unknown, wrong-horizon or contradictory hypothesis')
            if note['primary_evidence_id'] not in row['source_ids']:
                row['source_ids'].append(note['primary_evidence_id'])
        after = dict(source_ids=row['source_ids'], selected_hypotheses=row['selected_hypotheses'])
        if before != after:
            adjustments.append(dict(horizon=horizon, before=before, after=after))
    report = validate_final(canonical(value), source_ids, notes)
    if packet is not None:
        checks = final_attribution_checks(report, packet, notes)
        issues = [f'{h}:{issue}' for h, check in checks.items() for issue in check['issues']]
        if issues:
            raise ValueError('company_attribution: ' + ', '.join(issues))
    return report, adjustments


def invalid(reason):
    return {str(h): {'prediction': 'INVALID', 'reason': reason} for h in HORIZONS}


def deterministic(packet, condition):
    predictions, forecasts = {}, {}
    for h in HORIZONS:
        rows = history(packet, h)
        expected = None
        if condition == 'hold_price':
            expected, direction = 0.0, 'NON_UP'
        elif condition == 'always_up':
            direction = 'UP'
        elif condition == 'momentum_20':
            if len(rows) < 21:
                raise ValueError('Insufficient history for 20-session momentum')
            direction = 'UP' if rows[-1]['close'] > rows[-21]['close'] else 'NON_UP'
        else:
            signals = {s['payload']['component']: s['payload']['values']
                       for s in packet['evidence'] if s['kind'] == 'engine_signal'}
            if condition == 'engine_price_only':
                signals = dict(events={'event_bias': 0.0}, tech={'tech_bias': 0.0}, market={'macro_bias': 0.0})
            if set(signals) != {'events', 'tech', 'market'}:
                raise ValueError('Full engine requires frozen event/tech/macro inputs')
            f = forecast_from_inputs(packet['company_code'], h, timestamp(packet['as_of']), rows,
                                     signals['events'], signals['tech'], signals['market'])
            if f is None:
                raise ValueError('Insufficient engine history')
            forecasts[str(h)] = asdict(f)
            expected = f.expected_return
            direction = 'UP' if expected > 0 else 'NON_UP'
        predictions[str(h)] = dict(prediction=direction, reason='valid', expected_return=expected)
    return predictions, forecasts


def run_condition(packet, condition, generate, *, mock=False, development=False, final_token_limit=None):
    """generate(request) returns content, finish_reason and usage/cost receipts.

    The injected real transport must enforce its own HTTP call and monetary caps.
    No retry, retrieval, implicit resume or reuse is performed by this adapter.
    """
    validate_packet(packet, development=development)
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    if condition not in {c['id'] for c in protocol['conditions']}:
        raise ValueError('Unknown condition')
    settings = protocol['generation']
    if final_token_limit is not None and final_token_limit <= 0:
        raise ValueError('Positive final token cap required')
    started = time.perf_counter()
    effective, removed = without_news(packet) if condition == 'specialist_without_news_data' else (deepcopy(packet), [])
    full = model_packet(effective)
    result = dict(case_id=packet['id'], condition=condition, original_input_sha256=sha(packet),
                  effective_input_sha256=sha(full), removed_source_ids=removed, analyses=[], calls=[],
                  raw_final=None, final=None, failure=None, predictions=invalid('not_run'))

    def ask(phase, system, payload, schema, tokens):
        request = dict(model=settings['model_snapshot'], temperature=settings['temperature'], seed=settings['seed'],
            max_tokens=tokens, messages=[{'role': 'system', 'content': system},
                                         {'role': 'user', 'content': canonical(payload)}],
            response_format={'type': 'json_schema', 'json_schema': {'name': phase, 'strict': True, 'schema': schema}})
        call = dict(phase=phase, request=request, request_sha256=sha(request), payload_sha256=sha(payload), state='started')
        result['calls'].append(call)
        start = time.perf_counter()
        try:
            response = generate(deepcopy(request))
            call['response'] = deepcopy(response)
            if response.get('finish_reason') != 'stop' or not isinstance(response.get('content'), str):
                raise ValueError('Incomplete response')
            call['state'] = 'success'
            return response['content']
        except Exception as exc:
            call.update(state='failed', error_type=type(exc).__name__)
            raise
        finally:
            call['elapsed_seconds'] = time.perf_counter() - start

    try:
        if not condition.startswith(('single_', 'generic_', 'specialist_')):
            result['predictions'], result['forecasts'] = deterministic(packet, condition)
        else:
            roles = [] if condition == 'single_llm' else [r for r in ROLES
                        if not (condition == 'specialist_without_news_role' and r == 'news')]
            for role in roles:
                part = model_packet(effective, role)
                raw = ask('analysis_' + role, analyst_prompt('generic' if condition == 'generic_panel' else role),
                          part, Analysis.model_json_schema(), settings['initial_token_limit'])
                analysis = validate_analysis(raw, {s['id'] for s in part['evidence']})
                result['analyses'].append(dict(role=role, input_sha256=sha(part), analysis=analysis))
            result['analysis_attribution_checks'] = analysis_attribution_checks(effective, result['analyses'])
            excluded = {key: check['issues'] for key, check in result['analysis_attribution_checks'].items() if check['issues']}
            candidates = {f"{n['role']}:{h}":dict(primary_evidence_id=row['primary_evidence_id'], direction=row['direction'])
                          for n in result['analyses'] for h,row in n['analysis'].items() if f"{n['role']}:{h}" not in excluded}
            payload = dict(packet=full, independent_analyses=result['analyses'], hypothesis_ids=candidates,
                           excluded_hypotheses=excluded)
            rules = FINAL_RULES + ' hypothesis_ids에 실재하는 기간별 ID만 채택하고 그 primary_evidence_id를 source_ids에도 넣는다. '
            raw = ask('final', rules, payload, final_schema({s['id'] for s in full['evidence']}, result['analyses'], excluded), final_token_limit or settings['final_token_limit'])
            result['raw_final'] = raw
            draft, _ = bind_final(raw, {s['id'] for s in full['evidence']}, result['analyses'])
            result['final_attribution_checks'] = final_attribution_checks(draft, effective, result['analyses'])
            report, adjustments = bind_final(raw, {s['id'] for s in full['evidence']}, result['analyses'], effective)
            result['binding_adjustments'] = adjustments
            result['final'] = report
            result['predictions'] = {h: dict(prediction='INVALID' if row['direction'] == 'ABSTAIN' else row['direction'],
                reason='abstained' if row['direction'] == 'ABSTAIN' else 'valid') for h, row in report['predictions'].items()}
    except Exception as exc:
        result['failure'] = type(exc).__name__
        result['failure_detail'] = str(exc)[:2000]
        result['predictions'] = invalid('generation_or_validation_failed')
    result['wall_seconds'] = time.perf_counter() - started
    new_calls = [call for call in result['calls'] if not call.get('response', {}).get('reused')]
    result['api_duration_sum_seconds'] = sum(call['elapsed_seconds'] for call in new_calls) if not mock else 0
    result['callback_duration_sum_seconds'] = sum(c['elapsed_seconds'] for c in result['calls'])
    receipts = [call.get('response', {}) for call in new_calls]
    result['accounting'] = dict(generator='mock' if mock else 'injected_transport',
        callback_attempts=len(result['calls']), model_calls=0 if mock else len(new_calls), reused_calls=len(result['calls'])-len(new_calls),
        usage_complete=all(r.get('usage') is not None for r in receipts) if not mock else False,
        usage_receipts=[r.get('usage') for r in receipts],
        cost_usd=0 if mock or not receipts else (sum(r['cost_usd'] for r in receipts)
                  if all(r.get('cost_usd') is not None for r in receipts) else None),
        unknown_usage_calls=0 if mock else sum(r.get('usage') is None for r in receipts))
    return result


def run_comparison(packet, generate, *, mock=False, development=False, conditions=None):
    validate_packet(packet, development=development)
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    allowed = [c['id'] for c in protocol['conditions']]
    order = list(conditions) if conditions is not None else allowed
    if not order or len(set(order)) != len(order) or not set(order) <= set(allowed):
        raise ValueError('Invalid condition selection')
    random.Random(f"{protocol['generation']['seed']}:{packet['id']}").shuffle(order)
    start = time.perf_counter()
    results = [run_condition(packet, name, generate, mock=mock, development=development) for name in order]
    by_id = {r['condition']: r for r in results}
    deltas = {}
    for pair in protocol['evaluation']['paired_deltas']:
        a, b = pair.split('-')
        if a not in by_id or b not in by_id:
            continue
        deltas[pair] = {str(h): dict(left=by_id[a]['predictions'][str(h)]['prediction'],
                                    right=by_id[b]['predictions'][str(h)]['prediction'],
                                    correctness_transition=None) for h in HORIZONS}
    return dict(case_id=packet['id'], validation_scope=('development_reconstruction_unknown_price_availability'
                if development else 'structural_only_not_historical_certification'),
                execution='mock' if mock else 'injected_transport', predictive_evaluation=False,
                protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                concurrency=1, execution_order=order, wall_seconds=time.perf_counter()-start,
                results=results, paired_directions=deltas)


def mock_generate(request):
    """Positive plumbing control, not market predictions or model responses."""
    payload = json.loads(request['messages'][1]['content'])
    if request['response_format']['json_schema']['name'] == 'final':
        evidence = payload['packet']['evidence']
        prices = next(s['id'] for s in evidence if s['kind'] == 'price')
        # Always reserve judgment; a successful report need not invent a forecast.
        value = {'predictions': {str(h): dict(direction='ABSTAIN', source_ids=[prices], selected_hypotheses=[],
            observation='MOCK: 가격 입력 확인', hypothesis='MOCK: 추론 미실행',
            counter_condition='MOCK: 검사 전용', limitations='실제 모델 응답 아님') for h in HORIZONS}}
    else:
        value = {f'd{h}': dict(primary_evidence_id=None, supporting_evidence_ids=[], direction='판단 유보',
            mechanism='MOCK: 입력 분배 검사', counter_condition='MOCK: 검사 전용', missing='실제 모델 추론은 실행하지 않았다.') for h in HORIZONS}
    return dict(content=canonical(value), finish_reason='stop', usage=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    packet = json.loads(args.input.read_text(encoding='utf-8'))
    validate_packet(packet)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'input.json').write_text(canonical(packet), encoding='utf-8')
    result = run_comparison(packet, mock_generate, mock=True)
    result['code_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__), ROOT/'independent_reasoning.py', ROOT/'role_prompts.py',
         ROOT/'semiconductor_agent_design/app/forecast/price_forecast.py']}
    (args.output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    failures = sum(r['failure'] is not None for r in result['results'])
    print(canonical(dict(conditions=len(result['results']), mock_callbacks=sum(len(r['calls']) for r in result['results']),
                         failures=failures, model_calls=0, cost_usd=0, predictive_evaluation=False)))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
