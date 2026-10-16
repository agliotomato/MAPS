"""Source-bound financial directions, not a general financial fact verifier."""
import hashlib
import re

from evidence_claims import sentence_spans

FINANCIAL_SCOPE_RULES = (
    ' financial_facts는 원문에서 추출한 회사·부문·지표별 증감과 정확한 인용이다. '
    '연결 전체와 DS(반도체)·DX·파운드리 부문, 매출과 이익, 전분기와 전년동기 비교를 구분한다. '
    '전체 매출 감소나 부문 이익 악화를 부문 매출 감소로 바꾸지 않는다. '
    '같은 회사라도 부문별 매출 방향은 다를 수 있다. 원문에 없는 증감률·비교 기준을 채우지 않는다. '
    '보고된 과거 실적과 앞으로의 조건부 전망을 구분하며, 추출 목록은 전체 원문을 대체하지 않는다. '
)

SCOPES = {
    'DS': r'Device Solutions|\bDS\b|반도체\s*(?:부문|사업|사업부)|semiconductor\s+(?:division|business|segment)',
    'DX': r'Device eXperience|\bDX\b',
    'foundry': r'\bfoundry\b|파운드리',
    'consolidated': r'\b(?:the company|consolidated|total)\b|연결|전체|전사',
}
METRICS = re.compile(r'영업\s*이익|순이익|이익|수익성|매출(?:액)?|\brevenue\b|\b(?:operating |net )?profit\b|\bearnings\b', re.I)
DIRECTIONS = re.compile(r'증가|감소|늘었|늘어|줄었|줄어|\b(?:increase|decrease|decline|grow|growth|grew|fall|fell|rose|rise)\w*\b', re.I)


def period(text):
    patterns = (r'(20\d{2})\s*년\s*([1-4])\s*분기', r'(20\d{2})\s*[- ]?Q([1-4])')
    for pattern in patterns:
        if match := re.search(pattern, text, re.I):
            return f'{match[1]}Q{match[2]}'
    if match := re.search(r'\b(first|second|third|fourth) quarter(?: ended.*?)? (?:of )?(20\d{2})', text, re.I):
        return f"{match[2]}Q{('first', 'second', 'third', 'fourth').index(match[1].lower()) + 1}"
    return None


def comparison(text):
    if re.search(r'전년\s*동기|year.over.year|year.earlier', text, re.I):
        return 'YoY'
    if re.search(r'전분기|직전\s*분기|previous quarter|quarter.on.quarter', text, re.I):
        return 'QoQ'
    return None


def directions(text):
    """Explicit nearby metric/direction pairs; keep the exact sentence span."""
    # ponytail: bounded lexical coverage; unsupported wording stays unverified.
    # Expand with source-backed counterexamples, never infer missing financial facts.
    for start, end in sentence_spans(text):
        sentence = text[start:end]
        metrics = list(METRICS.finditer(sentence))
        for index, metric in enumerate(metrics):
            left = metrics[index-1].end() if index else 0
            right = metrics[index+1].start() if index+1 < len(metrics) else len(sentence)
            before, after = sentence[left:metric.start()], sentence[metric.end():right]
            # English "increase in revenue" belongs to the following metric.
            pre = re.search(r'(increase|decrease|decline|growth|rise|fall)\s+in\s*$', before, re.I)
            post = DIRECTIONS.search(after)
            if pre:
                word, tail = pre[1], after
            elif post and post.start() < 90 and not re.match(r'\s+in\b', after[post.end():], re.I):
                word, tail = post[0], after[post.end():]
            else:
                continue
            local = sentence[max(left, metric.start()-60):metric.start()] + before[-30:] + after
            if re.search(r'아니|않|없|미확인|\b(?:not|never|no)\b', local, re.I):
                continue
            # Do not turn a conditional/future counterexample into a reported result.
            if re.match(r'(?:할|한다|하|했|될|된다)?(?:다면|면| 경우)|\s*(?:할 경우|가능|전망|예상)', tail):
                continue
            if re.search(r'향후|앞으로|다음\s*분기|\b(?:if|may|might|could|would|will|expected|forecast)\b', local, re.I):
                continue
            prefix = sentence[:metric.start()]
            scopes = [(m.start(), scope) for scope, pattern in SCOPES.items() for m in re.finditer(pattern, prefix, re.I)]
            scope = max(scopes)[1] if scopes else None
            # "DS posted ... consolidated revenue" is DS, not the company total.
            segments = [pair for pair in scopes if pair[1] != 'consolidated']
            if segments and re.search(r'\bconsolidated\s*$', prefix, re.I):
                scope = max(segments)[1]
            yield dict(scope=scope, metric='revenue' if re.fullmatch(r'매출(?:액)?|revenue', metric[0], re.I) else 'profit',
                direction='increase' if re.search(r'증가|늘|increase|grow|grew|rose|rise', word, re.I) else 'decrease',
                period=period(sentence), comparison=comparison(sentence), quote=sentence, subject_prefix=prefix,
                span_start=start, span_end=end)


def financial_facts(sources):
    from independent_reasoning import company_scope
    facts = []
    for source in sources:
        if source.get('kind') == 'price':
            continue
        payload = source.get('payload', {})
        text = str(payload.get('text', source.get('text', '')) if isinstance(payload, dict) else payload)
        companies = company_scope(source)['companies']
        if len(companies) != 1:
            continue
        for claim in directions(text):
            # Missing comparison/segment remains unknown, never borrowed from another claim.
            facts.append(dict(claim, company=companies[0], period=claim['period'] or period(text),
                source_id=source.get('id', source.get('source_id')),
                source_sha256=hashlib.sha256(text.encode()).hexdigest(), verification='source_excerpt_only'))
    return facts


def financial_direction_issues(task, sources, *texts):
    from evidence_contract import COMPANY_ALIASES, mentions_company, target_company_aliases
    aliases = target_company_aliases(task)
    if not aliases:
        return []
    facts = financial_facts(sources)
    issues = []
    for text in texts:
        for claim in directions(text):
            named = [a[0] for a in COMPANY_ALIASES if mentions_company(claim['subject_prefix'], a)]
            company = named[0] if len(named) == 1 else aliases[0] if not named else None
            if not company or not claim['scope']:
                continue
            matches = [f for f in facts if (f['company'], f['scope'], f['metric']) ==
                (company, claim['scope'], claim['metric']) and
                (not claim['period'] or f['period'] == claim['period'])]
            if claim['comparison']:
                known = [f for f in matches if f['comparison'] == claim['comparison']]
                if matches and not known:
                    issues.append('financial_comparison_not_established')
                matches = known
            if matches and {f['direction'] for f in matches} == ({'increase', 'decrease'} - {claim['direction']}):
                issues.append(f"financial_direction_conflict:{claim['scope']}:{claim['metric']}")
    return sorted(set(issues))
