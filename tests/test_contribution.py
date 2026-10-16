"""Offline controls for information fairness, failures and unchanged engine math."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from benchmarks import contribution as c
from app.forecast import price_forecast as engine

FIXTURE = c.ROOT / 'benchmarks/contribution_fixture.json'


class ContributionTests(unittest.TestCase):
    def setUp(self):
        self.packet = json.loads(FIXTURE.read_text(encoding='utf-8'))

    def source(self, name):
        return next(s for s in self.packet['evidence'] if s['id'] == name)

    def test_cutoff_precision_and_outcome_window(self):
        c.validate_packet(self.packet)
        for when, precision in [('2026-04-01T00:00:00+00:00', 'timestamp'),
                                ('2026-04-01T00:01:00+00:00', 'timestamp'),
                                ('2026-04-01T00:00:00+09:00', 'date'),
                                ('2026-03-31T07:00:00', 'timestamp')]:
            packet = deepcopy(self.packet)
            packet['evidence'][-1].update(available_at=when, precision=precision)
            with self.subTest(when=when), self.assertRaises(ValueError):
                c.validate_packet(packet)
        s = self.source('signal-events')
        s.update(kind='event_outcome', payload=dict(window_end_available_at='2026-04-02T00:00:00Z',
                                                   computed_at='2026-03-31T07:00:00Z'))
        with self.assertRaises(ValueError):
            c.validate_packet(self.packet)

    def test_price_future_wrong_target_missing_sessions_and_metadata(self):
        changes = [lambda p: p.update(company_code='000660'),
                   lambda p: p['rows'][-1].update(trade_date='2026-04-01'),
                   lambda p: p['rows'][-1].update(available_at='2026-03-31T05:00:00Z'),
                   lambda p: p['rows'][-1].update(close=float('nan')),
                   lambda p: p['rows'].pop(), lambda p: p.pop('adjustment')]
        for change in changes:
            packet = deepcopy(self.packet)
            change(packet['evidence'][0]['payload'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.validate_packet(packet)
        self.packet['gold'] = {'7': 'UP'}
        with self.assertRaises(ValueError):
            c.run_comparison(self.packet, lambda *a: self.fail('Must reject before generation'), mock=True)

    def test_news_alias_and_transitive_removal(self):
        stripped, removed = c.without_news(self.packet)
        self.assertEqual(set(removed), {'news-1', 'copy-econ', 'derived-tech', 'signal-events', 'signal-tech'})
        self.assertEqual({s['id'] for s in stripped['evidence']},
                         {'price-1', 'tech-1', 'macro-1', 'disclosure-1', 'signal-macro'})
        self.assertEqual(len(self.packet['evidence']), 10)
        # A duplicate mislabeled with another origin is caught by exact payload too.
        self.source('copy-econ')['origin'] = 'wrong-independent-origin'
        self.assertIn('copy-econ', c.without_news(self.packet)[1])

    def test_lineage_missing_cycle_and_derived_timestamp(self):
        for parents in [['unknown'], ['derived-tech'], ['signal-tech']]:
            packet = deepcopy(self.packet)
            next(s for s in packet['evidence'] if s['id'] == 'derived-tech')['parents'] = parents
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                c.validate_packet(packet)
        self.source('derived-tech')['available_at'] = '2026-03-30T07:00:00Z'
        with self.assertRaises(ValueError):
            c.validate_packet(self.packet)
        self.source('derived-tech')['available_at'] = '2026-03-31T07:00:00Z'
        self.packet['limitations'] = ['news summary would leak through here']
        with self.assertRaises(ValueError):
            c.validate_packet(self.packet)

    def test_all_conditions_identical_final_information_and_isolated_partitions(self):
        before = c.sha(self.packet)
        run = c.run_comparison(self.packet, c.mock_generate, mock=True)
        results = {r['condition']: r for r in run['results']}
        self.assertEqual(len(results), 10)
        self.assertEqual(sum(len(r['calls']) for r in results.values()), 20)
        self.assertTrue(all(r['failure'] is None for r in results.values()))
        self.assertEqual(before, c.sha(self.packet))
        full = ['single_llm', 'generic_panel', 'specialist_panel', 'specialist_without_news_role']
        packets = [json.loads(results[n]['calls'][-1]['request']['messages'][1]['content'])['packet'] for n in full]
        self.assertEqual(len({c.canonical(p) for p in packets}), 1)
        self.assertEqual(len(results['specialist_without_news_role']['calls']), 4)
        self.assertIn('news-1', {s['id'] for s in packets[-1]['evidence']})
        self.assertNotIn('news', [n['role'] for n in results['specialist_without_news_role']['analyses']])
        for r in results.values():
            self.assertEqual(r['accounting']['model_calls'], 0)
            for call in r['calls'][:-1]:
                payload = json.loads(call['request']['messages'][1]['content'])
                role = call['phase'].removeprefix('analysis_')
                self.assertNotIn('independent_analyses', payload)
                self.assertTrue(all(s['role'] == role for s in payload['evidence']))
            if r['condition'] == 'specialist_without_news_data':
                for call in r['calls']:
                    text = call['request']['messages'][1]['content']
                    self.assertFalse(any(s in text for s in r['removed_source_ids']))
        for generic, specialist in zip(results['generic_panel']['calls'][:-1], results['specialist_panel']['calls'][:-1]):
            self.assertEqual(generic['payload_sha256'], specialist['payload_sha256'])
            self.assertEqual(generic['request']['max_tokens'], specialist['request']['max_tokens'])

    def test_failures_retained_no_retry_no_unknown_usage_as_zero(self):
        def fail(request):
            raise TimeoutError('synthetic transport failure')
        r = c.run_condition(self.packet, 'specialist_panel', fail)
        self.assertEqual(len(r['calls']), 1)
        self.assertEqual(r['accounting']['unknown_usage_calls'], 1)
        self.assertIsNone(r['accounting']['cost_usd'])
        self.assertEqual({v['prediction'] for v in r['predictions'].values()}, {'INVALID'})
        def malformed(request):
            return dict(content='{}', finish_reason='stop', usage={'prompt_tokens': 10, 'completion_tokens': 2}, cost_usd=.0001)
        r = c.run_condition(self.packet, 'single_llm', malformed)
        self.assertEqual(r['accounting']['cost_usd'], .0001)
        self.assertTrue(r['accounting']['usage_complete'])
        self.assertEqual(len(r['predictions']), 3)
        self.assertIsNotNone(r['raw_final'])

    def test_selected_hypotheses_are_bound_to_horizon_direction_and_source(self):
        analysis = {f'd{h}': dict(primary_evidence_id='price-1', supporting_evidence_ids=[], direction='Bullish',
            mechanism='관측 후 상승 가설', counter_condition='추세 반전', missing='인과 미검증') for h in c.HORIZONS}
        notes = [dict(role='market', analysis=analysis)]
        value = {'predictions': {str(h): dict(direction='UP', source_ids=['price-1'],
            selected_hypotheses=[f'market:d{h}'], observation='관측', hypothesis='가설',
            counter_condition='반전', limitations='미검증') for h in c.HORIZONS}}
        c.validate_final(c.canonical(value), {'price-1'}, notes)
        for changes in [dict(selected_hypotheses=['news:d7']), dict(selected_hypotheses=['market:d30']),
                        dict(direction='NON_UP'), dict(source_ids=['unknown']), dict(source_ids=[])]:
            bad = deepcopy(value)
            bad['predictions']['7'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                c.validate_final(c.canonical(bad), {'price-1'}, notes)
        with self.assertRaises(ValueError):
            c.validate_analysis(c.canonical(analysis), {'news-1'})

    def test_baselines_engine_ablation_and_insufficient_history(self):
        self.assertEqual({r['prediction'] for r in c.deterministic(self.packet, 'hold_price')[0].values()}, {'NON_UP'})
        for condition in ('always_up', 'momentum_20'):
            values, _ = c.deterministic(self.packet, condition)
            self.assertEqual({r['prediction'] for r in values.values()}, {'UP'})
            self.assertTrue(all(r['expected_return'] is None for r in values.values()))
        _, full = c.deterministic(self.packet, 'engine_full')
        _, price = c.deterministic(self.packet, 'engine_price_only')
        self.assertNotEqual(full['7']['expected_return'], price['7']['expected_return'])
        self.assertEqual(price['7']['signals'], dict(event_bias=0, tech_bias=0, macro_bias=0))
        p = self.source('price-1')['payload']
        p['rows'] = p['rows'][-20:]
        p['expected_sessions'] = p['expected_sessions'][-20:]
        r = c.run_condition(self.packet, 'momentum_20', c.mock_generate, mock=True)
        self.assertEqual(len(r['calls']), 0)
        self.assertEqual({v['prediction'] for v in r['predictions'].values()}, {'INVALID'})

    def test_engine_matches_nine_preserved_pre_refactor_outputs(self):
        golden = json.loads((c.ROOT/'benchmarks/contribution_engine_golden.json').read_text(encoding='utf-8'))
        for case in golden:
            h = case['horizon']
            with self.subTest(horizon=h, tech=case['tech'], macro=case['macro']):
                result = asdict(engine.forecast_from_inputs('005930', h, c.timestamp(self.packet['as_of']),
                    c.history(self.packet, h), {'event_bias': .2}, {'tech_bias': case['tech']}, {'macro_bias': case['macro']}))
                result.pop('created_at')
                self.assertEqual(result.pop('method'), 'scenario_v3_available_returns')
                expected = dict(case['expected'])
                expected.pop('method')  # Original v2 artifact stays intact; full-window math must match.
                self.assertEqual(result, expected)

    def test_production_wrapper_uses_shared_calculation_and_enrichment(self):
        with patch.object(engine, '_fetch_price_history', return_value=c.history(self.packet, 7)), \
             patch.object(engine, '_fetch_event_signals', return_value={'event_bias': .2}), \
             patch.object(engine, '_fetch_tech_signals', return_value={'tech_bias': .015}), \
             patch.object(engine, '_fetch_market_bias', return_value={'macro_bias': -.01}), \
             patch.object(engine, 'forecast_from_inputs', wraps=engine.forecast_from_inputs) as calculation, \
             patch.object(engine, 'enrich_signals_with_normalized_event', side_effect=lambda code, asof, s: s | {'enriched': True}):
            result = engine._forecast_for_company('005930', 7, c.timestamp(self.packet['as_of']))
            calculation.assert_called_once()
            self.assertTrue(result.signals['enriched'])
            self.assertEqual(result.expected_return, c.deterministic(self.packet, 'engine_full')[1]['7']['expected_return'])


if __name__ == '__main__':
    unittest.main()
