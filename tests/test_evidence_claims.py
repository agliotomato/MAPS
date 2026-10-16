"""Independent examples and counterexamples for narrow production claim rules."""
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from evidence_contract import planned_next_checks

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_claims import claim_for_quote, claim_issues, claim_policy, claim_explanation, claim_key
from evidence_contract import SourceReview, bind_report_type, quote_candidates, text_source, source_scope, check_report


def review(source, quote, explanation):
    claim = claim_for_quote(source, quote)
    material, status = claim_policy(claim) or ("other", "unknown")
    return SourceReview(source_id=source["id"], source_scope=source["source_scope"], quote=quote,
        claim=claim, material_type=material, claim_status=status,
        company_link="uncertain", interpretation=claim_explanation(claim, quote) if claim else explanation)


class ClaimTests(unittest.TestCase):
    def test_gift_is_reported_event_and_tax_or_price_is_not_gift_claim(self):
        source = text_source("independent:gift", "가상기업 주식 300주를 자녀에게 증여한다고 밝혔다.")
        row = review(source, source["text"], "가상기업 주식 300주 증여에 관한 보도다.")
        self.assertEqual(claim_issues(source, row), [])
        self.assertEqual((row.claim.kind, row.claim.epistemic_status), ("event", "reported"))
        row.material_type = "price_observation"
        self.assertIn("claim_type_status_mismatch", claim_issues(source, row))
        for text in ("주식 증여세 제도를 설명한다.", "증여받은 주식의 종가는 500원이었다."):
            self.assertIsNone(claim_for_quote(text_source("negative", text), text))

    def test_forecast_keeps_relative_period_without_inventing_observation_date(self):
        text = "올해 및 내년 성장률은 1.8% 및 2.2%로 전망된다."
        source = text_source("independent:growth", text, metadata={"published_date": "2025-08-01"})
        row = review(source, text, "올해 및 내년 성장률을 1.8% 및 2.2%로 전망한다.")
        self.assertEqual(claim_issues(source, row), [])
        self.assertEqual(row.claim.time_text, "올해 및 내년")
        self.assertEqual(row.claim.published_at, "2025-08-01")
        row.material_type, row.claim_status = "economic_observation", "confirmed"
        row.interpretation = "성장률 전망 2.2%를 달성했다."
        self.assertIn("claim_type_status_mismatch", claim_issues(source, row))
        self.assertIn("forecast_presented_as_realized", claim_issues(source, row))
        actual = "성장률은 전망보다 낮았으며 실제 1.5%를 기록했다."
        self.assertIsNone(claim_for_quote(text_source("actual", actual), actual))

    def test_fx_interpretation_cannot_borrow_other_passage_or_numbers(self):
        text = "원\n/\n달러 환율은 달러 약세로 하락하였다. 통화정책은 기준금리 인상을 검토한다."
        source = text_source("independent:fx", text)
        quote = quote_candidates(text)[0]
        row = review(source, quote, "원/달러 환율이 달러 약세로 하락했다는 설명이다.")
        self.assertEqual(claim_issues(source, row), [])
        row.interpretation = "통화정책은 기준금리 인상을 검토한다."
        self.assertIn("claim_subject_missing", claim_issues(source, row))
        self.assertIn("interpretation_topic_outside_quote:통화정책", claim_issues(source, row))
        row.interpretation = "환율이 100원 하락했다."
        self.assertIn("interpretation_number_outside_quote", claim_issues(source, row))
        row = review(source, quote, "환율이 하락했다.")
        row.claim.span_start += 1
        self.assertIn("claim_binding_mismatch", claim_issues(source, row))

    def test_metric_binds_value_unit_date_only_when_in_selected_quote(self):
        text = json.dumps({"current_value": 1301.5, "unit": "KRW/USD", "as_of": "2025-07-31"})
        source = text_source("independent:metric", text, kind="metric")
        row = review(source, text, "2025-07-31 원/달러 환율은 1,301.50 KRW/USD다.")
        self.assertEqual(claim_issues(source, row), [])
        self.assertEqual((row.claim.value, row.claim.unit, row.claim.time_text), (1301.5, "KRW/USD", "2025-07-31"))
        row.interpretation = "2025-07-31 원/달러 환율은 1,301.50 KRW/USD다."
        self.assertNotIn("interpretation_number_outside_quote", claim_issues(source, row))
        self.assertIsNone(claim_for_quote(source, '"unit": "KRW/USD"'))

    def test_mixed_document_does_not_force_one_type_and_context_is_not_observation(self):
        text = "가상기업 주식 10주를 증여한다. 종가는 100원이었다. 내년 성장률은 2%로 전망된다."
        source = text_source("mixed", text)
        claims = [claim_for_quote(source, q) for q in quote_candidates(text)]
        self.assertEqual([c.kind if c else None for c in claims], ["event", None, "forecast"])
        note = "Note: This report was prepared based on information collected before July 1."
        context = claim_for_quote(text_source("note", note), note)
        self.assertEqual(claim_policy(context), ("other", "not_applicable"))

    def test_wire_schema_rejects_wrong_type_and_restores_source_claim(self):
        source = text_source("arbitrary:id", "삼성전자 지분을 증여한다고 밝혔다.")
        bound = bind_report_type(lambda: [source], lambda: "삼성전자 분석")
        schema = bound.model_json_schema()
        local = {**schema["properties"]["source_reviews"]["properties"][source["id"]], "$defs": schema["$defs"]}
        row = review(source, source["text"], "삼성전자 지분 증여 보도다.").model_dump()
        row.pop("quote")
        row.pop("claim")
        row.pop("interpretation")
        row["quote_index"] = 0
        jsonschema.validate(row, local)
        wrong = dict(row, material_type="price_observation")
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(wrong, local)
        final = bound.model_validate(dict(answer="범위", price_evidence="unknown", source_reviews={source["id"]: row},
            judgment="판단 유보", confidence="Low", outlook_7d="판단 유보", outlook_14d="판단 유보",
            outlook_30d="판단 유보", next_checks=planned_next_checks([source])))
        claim = final.source_reviews[0].claim
        self.assertEqual(source["text"][claim.span_start:claim.span_end], final.source_reviews[0].quote)
        changed = dict(source, text=source["text"].replace("밝혔다", "설명했다"))
        self.assertNotEqual(claim.claim_id, claim_for_quote(changed, changed["text"]).claim_id)

    def test_actual_main_resolves_wire_claim_and_renders_checked_report(self):
        from test_production_report import production_fixtures, valid_final, replay_client, run_production
        fixtures = production_fixtures()
        name = "search_news_and_disclosures"
        payload = json.loads(fixtures[name])
        source = payload["evidence_handoff"]["sources"][0]
        source["text"] = "가상 A사 주식 300주를 자녀에게 증여한다고 밝혔다."
        source["source_scope"] = source_scope(len(source["text"]), len(source["text"]))
        fixtures[name] = json.dumps(payload, ensure_ascii=False)
        final = valid_final(fixtures).model_dump()
        for row in final["source_reviews"]:
            row.pop("quote")
            row.pop("claim")
            row.pop("interpretation")
            row["quote_index"] = 0
        final["source_reviews"][0].update(material_type="event_report", claim_status="reported")
        client = replay_client(SimpleNamespace(model_dump_json=lambda: json.dumps(final, ensure_ascii=False)))
        with TemporaryDirectory() as folder:
            result, calls = asyncio.run(run_production(Path(folder), fixtures, client))
            saved = next((Path(folder) / "reports").glob("*.md")).read_text(encoding="utf-8")
        claim = result.messages[-1].content.source_reviews[0].claim
        self.assertEqual(claim.predicate, "share_gift")
        self.assertIn(claim.claim_id, saved)
        self.assertIn("contract_passed", saved)

    def test_multiple_supported_topics_are_required_and_cannot_be_skipped(self):
        source = text_source("mixed", "내년 성장률은 2%로 전망된다. 원/달러 환율은 하락하였다. 주택가격은 보합이다.")
        bound = bind_report_type(lambda: [source])
        schema = bound.model_json_schema()
        local = {**schema["properties"]["source_reviews"]["properties"]["mixed"], "$defs": schema["$defs"]}
        selections = {}
        for i, quote in enumerate(quote_candidates(source["text"])):
            claim = claim_for_quote(source, quote)
            if claim is None:
                continue
            row = review(source, quote, "").model_dump()
            for field in ("quote", "claim", "interpretation"):
                row.pop(field)
            row["quote_index"] = i
            selections[claim_key(claim)] = row
        jsonschema.validate({"claims": selections}, local)
        with self.assertRaises(jsonschema.ValidationError):
            key = "exchange_rate"
            jsonschema.validate({"claims": {key: selections[key]}}, local)
        final = bound.model_validate(dict(answer="범위", price_evidence="unknown", source_reviews={"mixed": {"claims": selections}},
            judgment="판단 유보", confidence="Low", outlook_7d="판단 유보", outlook_14d="판단 유보",
            outlook_30d="판단 유보", next_checks=planned_next_checks([source])))
        self.assertEqual({r.claim.predicate for r in final.source_reviews}, {"growth_outlook", "exchange_rate"})
        self.assertEqual(len(final.source_reviews), 2)
        event = SimpleNamespace(type="ToolCallExecutionEvent", source="SearchExpert", content=[SimpleNamespace(
            name="search_news_and_disclosures", is_error=False,
            content=json.dumps({"evidence_handoff": {"version": 1, "sources": [source]}}))])
        message = SimpleNamespace(type="StructuredMessage", source="Integrator", content=final)
        checked = check_report([event, message])[0]
        self.assertNotIn("source_review_coverage_mismatch", checked["issues"])
        self.assertFalse(any("supported_claim_coverage_mismatch" in issue for issue in checked["issues"]))
        final.source_reviews.pop()
        self.assertIn("mixed: supported_claim_coverage_mismatch", check_report([event, message])[0]["issues"])


if __name__ == "__main__":
    unittest.main()
