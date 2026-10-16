"""Known real error plus scope/metric/period/conditional counterexamples."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
from benchmarks import contribution as c
from financial_claims import financial_facts, financial_direction_issues
import independent_reasoning as ir

SAVED = json.loads((ROOT/'tests/fixtures/financial.json').read_text(encoding='utf-8'))
PACKET = SAVED['packet']
SOURCE = next(s for s in PACKET['evidence'] if s['id'] == 'samsung_q2')
ISSUE = 'financial_direction_conflict:DS:revenue'
BAD = '삼성전자의 2025년 2분기 반도체 부문 매출 감소가 주가에 부담으로 작용했다.'
GOOD = '삼성전자의 2025년 2분기 연결 매출은 감소했지만 DS 부문 매출은 증가했다.'


class FinancialClaimsTests(unittest.TestCase):
    def test_source_claims_have_exact_spans_and_no_inferred_ds_comparison(self):
        facts = financial_facts([SOURCE])
        ds = next(f for f in facts if f['scope'] == 'DS' and f['metric'] == 'revenue')
        total = next(f for f in facts if f['scope'] == 'consolidated')
        self.assertEqual((ds['direction'], ds['period'], ds['comparison']), ('increase', '2025Q2', None))
        self.assertEqual((total['direction'], total['comparison']), ('decrease', 'QoQ'))
        for fact in facts:
            self.assertEqual(SOURCE['payload']['text'][fact['span_start']:fact['span_end']], fact['quote'])
        reversed_source = deepcopy(SOURCE)
        reversed_source['payload']['text'] = reversed_source['payload']['text'].replace('an increase in revenue', 'a decrease in revenue')
        self.assertEqual(financial_direction_issues('005930', [reversed_source], BAD), [])
        self.assertIn(ISSUE, financial_direction_issues('005930', [reversed_source], GOOD))

    def test_good_scope_profit_other_period_company_and_conditional_are_not_reversed(self):
        controls = [GOOD, '삼성전자 DS 매출은 증가했지만 연결 매출은 감소했다.',
            '삼성전자 DS 매출 증가에도 이익은 감소했다.',
            '삼성전자 DS 부문 매출이 감소한다면 주가에 부담이 될 수 있다.',
            '향후 삼성전자 DS 부문 매출 감소 가능성은 확인하지 못했다.',
            '삼성전자 DS 매출이 감소한 것은 아니다.',
            '삼성전자 2024년 2분기 DS 매출은 감소했다.',
            'SK하이닉스 반도체 부문 매출은 감소했다.']
        for text in controls:
            with self.subTest(text=text):
                self.assertEqual(financial_direction_issues('005930', [SOURCE], text), [])
        self.assertIn('financial_comparison_not_established', financial_direction_issues('005930', [SOURCE],
            '삼성전자 DS 매출은 전년 동기 대비 감소했다.'))

    def test_actual_preserved_bad_final_now_fails_both_text_fields_without_rewriting(self):
        result = SAVED
        raw = result['raw_final']
        final = json.loads(raw)
        sources = [s for s in PACKET['evidence'] if s['id'] in final['predictions']['30']['source_ids']]
        self.assertIn(ISSUE, ir.company_attribution_review('005930', sources, final['predictions']['30']['hypothesis'])['issues'])
        self.assertIn(ISSUE, ir.company_attribution_review('005930', sources, GOOD, BAD)['issues'])
        self.assertEqual(raw, c.canonical(final))
        with self.assertRaisesRegex(ValueError, ISSUE):
            c.bind_final(raw, {s['id'] for s in PACKET['evidence']}, result['analyses'], PACKET)

    def test_shared_production_candidates_and_final_guard(self):
        from test_independent_reasoning import fixture
        from test_production_report import valid_final
        from evidence_contract import text_source, forecast_basis_issues
        fixtures, _, _ = fixture()
        task = '삼성전자 2026-10-13 기준 7/14/30일(달력일) 독립 분석'
        # Operational text-source format uses the same source-bound checker.
        source = text_source('search:financial', SOURCE['payload']['text'], metadata={
            'company_codes':['005930'], 'published_at':'2026-10-12'})
        sources = [source]
        items, _ = ir.catalog(sources, task)
        primary = next(iter(items))
        row = dict(primary_evidence_id=primary, supporting_evidence_ids=[], direction='Bearish',
            mechanism=BAD, counter_condition='고부가 제품의 수요가 회복되어 수익성이 개선된다면 부담이 줄 수 있다.',
            missing='향후 기간별 실적 반영 시점과 신규 주문은 확인하지 못했다.')
        records = [dict(role='news', context=ir.context_for(sources, task),
            analysis={f'd{h}':deepcopy(row) for h in c.HORIZONS})]
        for role in ir.ROLES[1:]:
            withheld = dict(row, primary_evidence_id=None, direction='판단 유보', mechanism='담당 근거가 부족하다.')
            records.append(dict(role=role, context=ir.context_for(sources, task),
                analysis={f'd{h}':deepcopy(withheld) for h in c.HORIZONS}))
        self.assertEqual(ir.inference_candidates(sources, task, records), [])
        records[0]['analysis']['d7']['mechanism'] = GOOD + ' 연결 매출 둔화가 주문 기대를 낮춘다면 단기 부담이 될 수 있다.'
        bases = ir.inference_candidates(sources, task, records)
        self.assertEqual(len(bases), 1)
        final = valid_final(fixtures)
        final.judgment = final.outlook_7d = 'Bearish'
        final.forecast_basis = {'7':ir.InferredForecastBasis.model_validate(bases[0])}
        self.assertEqual(forecast_basis_issues(final, sources, task, records), [])
        final.forecast_basis['7'].mechanism = BAD
        self.assertTrue(any(ISSUE in error for error in forecast_basis_issues(final, sources, task, records)))


if __name__ == '__main__':
    unittest.main()
