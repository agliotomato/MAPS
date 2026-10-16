import asyncio
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from jsonschema import Draft202012Validator
from autogen_core.models import CreateResult, RequestUsage
from autogen_core import FunctionCall
from autogen_ext.models.replay import ReplayChatCompletionClient
from evidence_contract import (bind_report_type, forecast_candidates, forecast_basis_issues,
    check_report, checked_report_markdown, text_source, ReportDecision)
import independent_reasoning as ir
from agent_team import TOOL_NAMES

TASK = '삼성전자 2026-10-13 기준 7/14/30일(달력일) 독립 분석'


def fixture():
    from test_production_report import production_fixtures
    fixtures = production_fixtures()
    for name in fixtures:
        payload = json.loads(fixtures[name])
        for source in payload['evidence_handoff']['sources']:
            source['metadata']['published_at'] = '2026-10-12'
        fixtures[name] = json.dumps(payload, ensure_ascii=False)
    payload = json.loads(fixtures[TOOL_NAMES[0]])
    payload['evidence_handoff']['sources'].append(text_source('search:controlled',
        '삼성전자는 신규 제품 출하를 시작했다고 발표했다.', metadata={'published_at': '2026-10-12'}))
    fixtures[TOOL_NAMES[0]] = json.dumps(payload, ensure_ascii=False)
    sources = [s for raw in fixtures.values() for s in json.loads(raw)['evidence_handoff']['sources']]
    items, _ = ir.catalog(sources, TASK)
    records = []
    for role in ir.ROLES:
        primary = ir.primary_ids(items, role, TASK)
        rows = {}
        for days in (7, 14, 30):
            active = role == 'news' and days == 7
            rows[f'd{days}'] = dict(primary_evidence_id=primary[0] if active else None,
                supporting_evidence_ids=[], direction='Bullish' if active else '판단 유보',
                mechanism='출하가 실제 판매 증가로 이어지면 단기 기대에 영향을 줄 수 있다.' if active else '기간 근거 부족.',
                counter_condition='실제 판매로 이어지지 않을 경우.', missing='매출 효과와 시장 반영 미확인.')
        records.append(dict(role=role, context=ir.context_for(sources, TASK), analysis=rows))
    return fixtures, sources, records


def wire_for(fixtures, sources, records):
    from test_production_report import valid_final
    value = valid_final(fixtures).model_dump()
    value.pop('forecast_basis')
    value.pop('next_checks')
    value['decision'] = {key: value.pop(key) for key in ReportDecision.model_fields}
    value['decision'].update(evidence_status='forecast_basis_identified', judgment='Bullish', confidence='Low')
    for days in (7, 14, 30):
        value['decision'][f'outlook_{days}d'] = dict(direction='Bullish' if days == 7 else '판단 유보',
                                                   basis_index=0 if days == 7 else None)
    return value


class IndependentReasoningTests(unittest.TestCase):
    def test_unquoted_inference_survives_with_exact_source_and_no_period_extension(self):
        fixtures, sources, records = fixture()
        self.assertEqual(forecast_candidates(sources, TASK), [])
        wire = wire_for(fixtures, sources, records)
        bound = bind_report_type(lambda: sources, lambda: TASK, lambda: records)
        final = bound.model_validate(wire)
        self.assertEqual(final.forecast_basis['7'].basis_kind, 'model_inference')
        self.assertEqual(final.outlook_14d, '판단 유보')
        self.assertEqual(forecast_basis_issues(final, sources, TASK, records), [])
        self.assertIn('outlook_7d: forecast_basis_mismatch', forecast_basis_issues(final, sources, TASK))
        for key, value in [('horizon_days', 30), ('quote', 'invented'), ('as_of', '2026-10-14'),
                           ('horizon_unit', 'trading_sessions'), ('target', 'SK하이닉스')]:
            changed = final.model_dump()
            changed['forecast_basis']['7'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bound.model_validate(changed)

    def test_future_unknown_missing_role_and_modified_context_rejected(self):
        _, sources, records = fixture()
        self.assertIsNone(ir.target(TASK.replace('달력일', '거래일')))
        self.assertIsNone(ir.target('삼성전자 분석'))
        future = text_source('search:future', '삼성전자 실적 발표.', metadata={'published_at': '2026-10-14'})
        unknown = text_source('search:unknown', '삼성전자 실적 발표.')
        items, excluded = ir.catalog(sources + [future, unknown], TASK)
        self.assertIn('search:future', excluded)
        self.assertIn('search:unknown', excluded)
        for rows in (records[:-1], records[::-1]):
            with self.assertRaises(ValueError):
                ir.inference_candidates(sources, TASK, rows)
        changed = copy.deepcopy(records)
        changed[0]['analysis']['d7']['primary_evidence_id'] = 'search:future@0'
        with self.assertRaises(ValueError):
            ir.inference_candidates(sources, TASK, changed)
        changed = copy.deepcopy(records)
        changed[0]['context']['input_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            ir.inference_candidates(sources, TASK, changed)
        empty = ir.bind_analysis({}, 'news', TASK)
        self.assertFalse(Draft202012Validator(empty.model_json_schema()).is_valid(records[0]['analysis']))

    def test_live_failure_guards_and_reference_deduplication(self):
        _, sources, records = fixture()
        items, _ = ir.catalog(sources, TASK)
        row = ir.Hypothesis.model_validate(records[0]['analysis']['d7'])
        for text in ('/', 'null', 'None'):
            row.missing = text
            self.assertIn('unspecified_limitations', ir.admission_issues('news', row, items, sources))
        row.missing = '시장 반응을 확인하지 못했다.'
        row.direction, row.mechanism = 'Neutral', '영향이 불확실하다.'
        self.assertIn('uncertainty_is_not_neutral', ir.admission_issues('news', row, items, sources))
        row.supporting_evidence_ids = [row.primary_evidence_id, row.primary_evidence_id]
        value = dict(records[0]['analysis'], d7=row.model_dump())
        parsed = ir.bind_analysis(ir.role_catalog(items, 'news'), 'news', TASK).model_validate(value)
        self.assertEqual(parsed.d7.supporting_evidence_ids, [])
        self.assertEqual(parsed.d7.mechanism, row.mechanism)
        price = text_source('market:test', json.dumps([
            {'Date':'2026-10-12','AssetCode':'005930','Close':100},
            {'Date':'2026-10-13','AssetCode':'005930','Close':99}]), kind='price')
        price_items, _ = ir.catalog([price], TASK)
        row.primary_evidence_id = next(iter(price_items))
        row.direction = 'Bearish'
        for wording in ('30일 동안의 하락 추세', '14일 기준으로 하락', '7일간 하락'):
            row.mechanism = wording
            self.assertIn('claimed_price_history_exceeds_supplied_window',
                          ir.admission_issues('market', row, price_items, [price]))

    def test_actual_main_independent_messages_and_report_writer(self):
        from test_production_report import run_production
        fixtures, sources, records = fixture()
        responses = [CreateResult(finish_reason='function_calls', cached=False,
            usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
            content=[FunctionCall(id=name, name=name, arguments=json.dumps(args))])
            for name, args in zip(TOOL_NAMES, ({'query':'삼성전자'}, {'question':'삼성전자'},
                                             {'collect_data':False}, {'query':'삼성전자'}))]
        responses += [json.dumps(r['analysis'], ensure_ascii=False) for r in records]
        responses.append(json.dumps(wire_for(fixtures, sources, records), ensure_ascii=False))
        client = ReplayChatCompletionClient(responses, model_info={'vision':False, 'function_calling':True,
            'json_output':True, 'structured_output':True, 'family':'unknown'})
        with TemporaryDirectory() as folder:
            result, calls = asyncio.run(run_production(Path(folder), fixtures, client, task=TASK, independent=True))
            self.assertEqual(calls, list(TOOL_NAMES))
            check, final = check_report(result.messages)
            self.assertEqual(check['issues'], [])
            self.assertEqual(final.outlook_7d, 'Bullish')
            notes = ir.records_from_messages(result.messages)
            self.assertEqual(notes, records)
            text = next(Path(folder).glob('reports/report_*.md')).read_text(encoding='utf-8')
            self.assertIn('7달력일 독립 추론', text)
            self.assertIn('반대 조건', text)
            self.assertEqual(text.count('### 독립 분석 기록'), 4)
            missing = [m for m in result.messages if m.source != ir.NOTE_PREFIX + 'news']
            self.assertIn('invalid_independent_analyses', check_report(missing)[0]['issues'])
            self.assertNotIn('판단: Bullish', checked_report_markdown(missing))


if __name__ == '__main__':
    unittest.main()
