"""Technical excerpts and unsupported interpretations: independent counterexamples."""
import json
from pathlib import Path
import unittest
from evidence_contract import planned_next_checks
from types import SimpleNamespace

import jsonschema
from evidence_claims import claim_for_quote
from evidence_contract import (ProductionReport, bind_report_type, check_report, quote_candidates,
    review_explanation, review_policy, text_source)


def bind(source):
    bound = bind_report_type(lambda: [source])
    quote = quote_candidates(source["text"])[0]
    material, status = review_policy(source, quote)
    row = dict(source_id=source["id"], source_scope=source["source_scope"],
               quote_index=0, material_type=material, claim_status=status, company_link="uncertain")
    schema = bound.model_json_schema()
    local = {**schema["properties"]["source_reviews"]["properties"][source["id"]], "$defs": schema["$defs"]}
    jsonschema.validate(row, local)
    final = bound.model_validate(dict(answer="범위", price_evidence="unknown", source_reviews={source["id"]: row},
        judgment="판단 유보", confidence="Low", outlook_7d="판단 유보", outlook_14d="판단 유보",
        outlook_30d="판단 유보", next_checks=planned_next_checks([source])))
    return final, row, local


class TechnicalGroundingTests(unittest.TestCase):
    def test_followups_are_code_generated_and_cannot_be_empty_in_saved_report(self):
        final, _, _ = bind(text_source("unknown", "Unclassified source text."))
        final_data = final.model_dump()
        final_data["next_checks"] = []
        with self.assertRaises(ValueError):
            type(final).model_validate(final_data)
        self.assertNotIn("next_checks", type(final).model_json_schema()["properties"])
        schema = ProductionReport.model_json_schema()
        schema = {**schema["properties"]["next_checks"], "$defs": schema["$defs"]}
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate([], schema)

    def test_runon_results_keep_separate_forward_looking_clause(self):
        text = "Reports revenues of 8.0 billion USD from memory sales By developing new devices, the company will serve other markets."
        source = text_source("runon", text)
        quote = quote_candidates(text)[0]
        self.assertNotIn("By developing", quote)
        self.assertEqual(claim_for_quote(source, quote).predicate, "financial_results")

    def test_new_negation_forecast_and_section_boundary_examples(self):
        cases = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/technical_counterexamples.json").read_text(encoding="utf-8"))
        for case in cases:
            source = text_source(case["id"], case["text"], metadata={"source_type": case.get("source_type", "news")})
            quotes = [case["quote"]] if "quote" in case else quote_candidates(source["text"])
            if case["id"] == "denial_outside_window":
                self.assertTrue(all("denied" in quote for quote in quotes))
            # Negative documents must not yield an asserted claim through another window.
            if case["expected"] is not None:
                quotes = quotes[:1]
            for quote in quotes:
                with self.subTest(case=case["id"], quote=quote):
                    claim = claim_for_quote(source, quote)
                    self.assertEqual(claim.predicate if claim else None, case["expected"])

    def test_technical_numbers_and_financial_results_are_not_stock_prices(self):
        cases = [
            ("HBM memory delivers 15x bandwidth at a cost of 200 USD.", "paper", "technical_document", "technical_description"),
            ("The company has begun mass production of a 96GB memory module.", "news", "event_report", "production_announcement"),
            ("Example Inc. will present its earnings on Apr. 25, 2025.", "news", "event_report", "earnings_schedule"),
            ("Reports revenues of 12.3 trillion won and operating profit of 4.5 trillion won.", "news", "financial_report", "financial_results"),
            ("2025-04-25 주가는 12000원에 마감했다.", "news", "price_observation", "stock_price"),
        ]
        for text, kind, material, predicate in cases:
            with self.subTest(predicate=predicate):
                source = text_source("independent", text, metadata={"source_type": kind})
                final, row, schema = bind(source)
                review = final.source_reviews[0]
                self.assertEqual(review.material_type, material)
                self.assertEqual(review.claim.predicate, predicate)
                self.assertIn(" ".join(text.strip(".").split()), review.interpretation)
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.validate(dict(row, interpretation="삼성전자 주가 상승이 확정됐다."), schema)
                if material != "price_observation":
                    with self.assertRaises(jsonschema.ValidationError):
                        jsonschema.validate(dict(row, material_type="price_observation"), schema)

    def test_references_are_not_paper_results_and_do_not_swallow_body(self):
        body = "HBM memory delivers improved bandwidth in our experiments.\n"
        refs = "[1] One, Memory design, 2020.\n[2] Two, HBM, 2021.\n[3] Three, Architecture, 2022."
        source = text_source("paper", body + refs, metadata={"source_type": "paper"})
        self.assertEqual(claim_for_quote(source, body.strip()).predicate, "technical_description")
        self.assertEqual(claim_for_quote(source, refs[:refs.index("\n")]).predicate, "bibliography")
        clipped = "Core-memory control. IEEE Transactions, 2020.\n" + refs
        source = text_source("clipped", clipped, metadata={"source_type": "paper"})
        self.assertEqual(claim_for_quote(source, quote_candidates(clipped)[0]).predicate, "bibliography")

    def test_empty_price_source_does_not_assert_price_observations(self):
        for kind in ("price", "document"):
            source = text_source("empty", "", kind=kind, metadata={"title": "HBM forecast price 100"})
            final, _, _ = bind(source)
            row = final.source_reviews[0]
            self.assertEqual((row.material_type, row.claim_status), ("insufficient_information", "unknown"))
            self.assertIsNone(row.claim)

    def test_unknown_excerpts_keep_exact_content_without_guessed_meaning(self):
        source = text_source("unknown", "<p>Asterism&#160;is described here.</p>")
        final, _, _ = bind(source)
        row = final.source_reviews[0]
        self.assertEqual(row.quote, source["text"])
        self.assertEqual((row.material_type, row.claim_status), ("other", "unknown"))
        self.assertIn("Asterism is described here", row.interpretation)
        self.assertIn("의미 검수 필요", row.interpretation)

    def test_checker_rejects_borrowed_interpretation_even_without_known_claim(self):
        for text in ("Firms reported significant health care and insurance cost pressures.",
                     "Office furnishings were discussed by respondents."):
            source = text_source("economy", text)
            final, _, _ = bind(source)
            event = SimpleNamespace(type="ToolCallExecutionEvent", source="EconomyExpert", content=[SimpleNamespace(
                name="analyze_macro_economy", is_error=False,
                content=json.dumps({"evidence_handoff": {"version": 1, "sources": [source]}}))])
            message = SimpleNamespace(type="StructuredMessage", source="Integrator", content=final)
            self.assertFalse(any("explanation_mismatch" in issue for issue in check_report([event, message])[0]["issues"]))
            final.source_reviews[0].interpretation = "반도체 제조업 수요가 늘어 삼성전자 실적이 개선된다."
            self.assertIn("economy: source_explanation_mismatch", check_report([event, message])[0]["issues"])

    def test_abbreviations_html_and_metric_units_survive(self):
        text = "<p>Example Inc. will present earnings on Apr. 25, 2025. Next sentence.</p>"
        quote = quote_candidates(text)[0]
        self.assertIn("Inc. will present", quote)
        self.assertIn("Apr. 25", quote)
        self.assertIn(quote, text)
        metric = json.dumps(dict(current_value=99.4, unit="INDEX", as_of="2025-06-01"))
        source = text_source("economy:metric:manufacturing", metric, kind="metric")
        final, _, _ = bind(source)
        claim = final.source_reviews[0].claim
        self.assertEqual((claim.value, claim.unit, claim.time_text), (99.4, "INDEX", "2025-06-01"))
        self.assertNotIn("삼성전자", review_explanation(source, metric))


if __name__ == "__main__":
    unittest.main()
