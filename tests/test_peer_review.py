import asyncio
import copy
import json
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from peer_review import HORIZONS, review_round, validate_review
from test_independent_reasoning import fixture, TASK, wire_for
import independent_reasoning as ir


def review_for(record, challenge=False):
    return {h:dict(verdict='challenge' if challenge and h=='d7' else 'supported', field='mechanism',
        claim_quote=row['mechanism'], source_id='source', source_quote='원문 사실',
        reason='관측된 사실과 미래 전망은 구분해야 합니다.', proposed_correction='조건부 가설로 명시합니다.')
        for h,row in record['analysis'].items()}


def reply_for(record, change=False):
    value = copy.deepcopy(record['analysis'])
    if change:
        value['d7']['mechanism'] += ' 미래 효과는 아직 확인하지 못한 가설이다.'
    return dict(analysis=value, decisions={h:dict(action='revise' if change and h=='d7' else 'retain',
        reason='원문과 대조한 뒤 근거가 확인되는 내용만 유지했습니다.') for h in HORIZONS})


class PeerReviewTests(unittest.TestCase):
    def test_frozen_review_then_revision_preserves_other_roles_and_originals(self):
        _, _, records = fixture()
        original = copy.deepcopy(records)
        events = []
        async def generate(phase, rules, payload, schema, tokens):
            events.append(phase)
            author = next(r for r in original if r['role']==payload['author'])
            if phase.startswith('review_'):
                self.assertEqual(payload['author_analysis'], author['analysis'])
                return review_for(author, True)
            return reply_for(author, True)
        revised, trace = asyncio.run(review_round(records, {'source':'원문 사실을 그대로 인용.'}, {}, generate,
            lambda role, value:ir.Analysis.model_validate(value).model_dump()))
        self.assertEqual(trace['status'], 'completed')
        self.assertEqual(events, ['review_news','review_semiconductor','revise_news','revise_semiconductor'])
        self.assertEqual(records, original)
        self.assertEqual(revised[1:3], original[1:3])
        self.assertNotEqual(revised[0], original[0])
        self.assertNotEqual(revised[3], original[3])

    def test_fabricated_claim_source_and_missing_quote_rejected(self):
        author = fixture()[2][0]
        for changes in ({'claim_quote':'없는 주장'}, {'source_quote':'없는 인용'},
                        {'source_id':'없는 출처'}, {'source_id':None,'source_quote':''}):
            review = review_for(author, True)
            review['d7'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_review(review, author, {'source':'원문 사실'})

    def test_false_retention_and_unrequested_changes_rollback_entire_round(self):
        records = fixture()[2]
        for challenged in (True, False):
            async def generate(phase, rules, payload, schema, tokens):
                author = next(r for r in records if r['role']==payload['author'])
                if phase.startswith('review_'):
                    return review_for(author, challenged)
                result = reply_for(author, True)
                if challenged:
                    result['decisions']['d7']['action'] = 'retain'
                return result
            revised, trace = asyncio.run(review_round(records, {'source':'원문 사실'}, {}, generate,
                lambda role, value: value))
            self.assertEqual(trace['status'], 'failed_originals_retained')
            self.assertEqual(revised, records)
            self.assertEqual(len(trace['revisions']), 1)

    def test_mislabeled_exact_quote_binds_only_to_unique_original_field(self):
        author = fixture()[2][0]
        raw = review_for(author, True)
        raw['d7']['field'] = 'counter_condition'
        bound = validate_review(raw, author, {'source':'원문 사실'})
        self.assertEqual(bound['d7']['field'], 'mechanism')
        self.assertEqual(raw['d7']['field'], 'counter_condition')
        ambiguous = copy.deepcopy(author)
        ambiguous['analysis']['d7']['missing'] = ambiguous['analysis']['d7']['mechanism']
        with self.assertRaises(ValueError):
            validate_review(raw, ambiguous, {'source':'원문 사실'})

    def test_cancel_is_not_silently_converted_to_success(self):
        async def generate(*args):
            raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(review_round(fixture()[2], {}, {}, generate, lambda r,v:v))

    def test_actual_main_flag_review_trace_notes_and_checked_report(self):
        from autogen_core.models import CreateResult, RequestUsage
        from autogen_core import FunctionCall
        from autogen_ext.models.replay import ReplayChatCompletionClient
        from agent_team import TOOL_NAMES
        from evidence_contract import check_report
        from test_production_report import run_production
        fixtures, sources, records = fixture()
        responses = [CreateResult(finish_reason='function_calls', cached=False,
            usage=RequestUsage(prompt_tokens=0, completion_tokens=0), content=[FunctionCall(id=n,name=n,
            arguments=json.dumps(a))]) for n,a in zip(TOOL_NAMES, ({'query':'삼성전자'}, {'question':'삼성전자'},
                {'collect_data':False}, {'query':'삼성전자'}))]
        responses += [json.dumps(r['analysis'], ensure_ascii=False) for r in records]
        items,_ = ir.catalog(sources,TASK)
        sid = next(k for k in items if k.startswith('search:'))
        for role in ('news','semiconductor'):
            review = review_for(next(r for r in records if r['role']==role))
            for row in review.values():
                row.update(source_id=sid, source_quote=items[sid]['quote'])
            responses.append(json.dumps(review, ensure_ascii=False))
        responses += [json.dumps(reply_for(next(r for r in records if r['role']==role)), ensure_ascii=False)
                      for role in ('news','semiconductor')]
        responses.append(json.dumps(wire_for(fixtures,sources,records), ensure_ascii=False))
        client = ReplayChatCompletionClient(responses, model_info={'vision':False,'function_calling':True,
            'json_output':True,'structured_output':True,'family':'unknown'})
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        with TemporaryDirectory() as folder, patch.object(sys,'argv',['main.py','--peer-review']), \
             patch.object(socket.socket,'connect',side_effect=AssertionError('Network forbidden')):
            result,calls = loop.run_until_complete(run_production(Path(folder),fixtures,client,task=TASK,independent=True))
            self.assertEqual(calls,list(TOOL_NAMES))
            self.assertEqual(ir.records_from_messages(result.messages),records)
            self.assertEqual(check_report(result.messages)[0]['issues'],[])
            trace = json.loads(next(m.content for m in result.messages if m.source=='PeerReviewTrace'))
            self.assertEqual(trace['status'],'completed')
            report = next(Path(folder).glob('reports/report_*.md')).read_text(encoding='utf8')
            self.assertIn('PeerReviewTrace',report)
            self.assertIn('IndependentDraft_news',report)
            self.assertEqual(report.count('### 독립 분석 기록'),4)

    def test_paired_final_uses_same_source_model_rules_and_token_cap(self):
        from benchmarks import contribution as c
        saved = json.loads((Path(__file__).parent/'fixtures/paired.json').read_text(encoding='utf8'))
        packet, notes = saved['packet'], saved['analyses']
        requests = []
        def generate(body):
            phase = body['response_format']['json_schema']['name']
            if phase.startswith('analysis_'):
                note = next(n for n in notes if n['role']==phase.removeprefix('analysis_'))
                return dict(content=c.canonical(note['analysis']),finish_reason='stop',reused=True)
            requests.append(body)
            return dict(content=saved['raw_final'],finish_reason='stop',usage={},cost_usd=0)
        for _ in range(2):
            result = c.run_condition(packet,'specialist_panel',generate,development=True,final_token_limit=2200)
            self.assertEqual(result['analyses'],notes)
        self.assertEqual(requests[0],requests[1])
        self.assertEqual(requests[0]['max_tokens'],2200)


if __name__ == '__main__':
    unittest.main()
