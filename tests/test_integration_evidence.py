"""Offline regression checks for evidence handoff, with DB/model boundaries replaced."""
import importlib.util
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "economy_agent/src"))

import pandas as pd
from market_agent import market_agent as market
from integration_tools import (analyze_macro_economy, analyze_market_technicals, report_markdown,
                               search_news_and_disclosures, analyze_semiconductor_tech)


def load_economy_file(relative, replacements):
    # Execute the actual module while isolating DB/LLM package-level imports.
    path = ROOT / "economy_agent/src/macro_agent" / relative
    spec = importlib.util.spec_from_file_location("offline_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        # Windows asyncio creates a loopback socket pair for its own wakeups.
        self.loop = asyncio.new_event_loop()
        self.addCleanup(self.loop.close)
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        network.start()
        self.addCleanup(network.stop)

    def macro_result(self, result):
        cls = Mock()
        cls.return_value.invoke.return_value = result
        with patch.dict(sys.modules, {"macro_agent.agent": SimpleNamespace(MacroAnalysisAgent=cls)}):
            return json.loads(analyze_macro_economy("가상 기업의 경제 위험"))

    def market_result(self, reports, collect_data=False):
        with patch.object(market, "MarketAgent") as cls:
            cls.return_value.run.return_value = {"report": {"json": {
                "generated_at": "2026-09-30 09:00:00", "reports": reports,
            }}}
            result = json.loads(analyze_market_technicals(collect_data))
            cls.assert_called_once_with(use_gpt=False)
            cls.return_value.run.assert_called_once_with(collect_data=collect_data, max_pages=50)
        return result

    def test_market_source_survives_features_tool_and_saved_report(self):
        dates = pd.date_range("2026-07-20", periods=9)
        price = pd.DataFrame({"Date": dates, "AssetCode": "005930", "AssetName": "가상 기업",
                              "Market": "TEST", "Source": "SyntheticPrice", "Close": range(100, 109),
                              "Volume": 1000})
        flow = pd.DataFrame([{"Date": day, "AssetCode": "005930", "AssetName": "가상 기업",
                              "Market": "TEST", "Source": "SyntheticFlow", "InvestorType": who,
                              "NetBuyVolume": 100} for day in dates for who in ("Foreign", "Institution")])
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.object(market, "ensure_output_dir"))
            for name in ("PRICE_FILE", "FLOW_FILE", "FEATURE_FILE", "REPORT_JSON_FILE", "REPORT_TABLE_FILE"):
                stack.enter_context(patch.object(market, name, Path(tmp) / getattr(market, name).name))
            price.to_csv(market.PRICE_FILE, index=False)
            flow.to_csv(market.FLOW_FILE, index=False)
            payload = json.loads(analyze_market_technicals())
            features = pd.read_csv(market.FEATURE_FILE, dtype={"AssetCode": str})
            report = json.loads(market.REPORT_JSON_FILE.read_text(encoding="utf-8"))["reports"][0]
        self.assertEqual(features.iloc[-1]["FlowSource"], "SyntheticFlow")
        actual = payload["reports"][0]
        self.assertEqual(actual["code"], "005930")
        self.assertEqual(actual["as_of_date"], "2026-07-28")
        self.assertEqual(actual["data_sources"][0]["provider"], "SyntheticPrice")
        self.assertEqual(actual["data_sources"][1]["provider"], "SyntheticFlow")
        for key in ("evidence", "limitations", "risk_flags"):
            self.assertEqual(actual[key], report[key])
        self.assertEqual(payload["data_mode"], "cached")

        from autogen_agentchat.messages import TextMessage, ToolCallExecutionEvent
        from autogen_core.models import FunctionExecutionResult
        original = json.dumps(payload, ensure_ascii=False)
        event = ToolCallExecutionEvent(source="MarketExpert", content=[FunctionExecutionResult(
            name="analyze_market_technicals", call_id="test-1", is_error=False, content=original)])
        rendered = report_markdown("개발 점검", [event, TextMessage(
            source="Integrator", content="날짜를 생략한 요약")], "2026-09-30T09:00:00+09:00")
        record = json.loads(rendered.split("```json\n", 1)[1].split("\n```", 1)[0])
        self.assertEqual(record["content"], original)
        self.assertIn("2026-07-28", rendered)
        self.assertIn("날짜를 생략한 요약", rendered)

    def test_market_missing_metadata_empty_and_failed_results(self):
        row = pd.Series({"Date": pd.Timestamp("2026-07-28"), "AssetCode": "TEST",
                         "AssetName": "가상 기업", "Market": "TEST"})
        report = market.MarketReporter().generate_report_for_asset(row)
        self.assertIsNone(report["data_sources"][0]["provider"])
        self.assertIsNone(report["data_sources"][1]["as_of_date"])
        self.assertEqual(report["flow_view"]["view"], "NotAvailable")
        self.assertTrue(report["limitations"])
        refreshed = self.market_result([report], collect_data=True)
        self.assertEqual(refreshed["analysis_status"], "partial")
        self.assertEqual(refreshed["reports"][0]["analysis_status"], "partial")
        self.assertEqual(refreshed["data_mode"], "refresh_requested")
        self.assertEqual(refreshed["reports"][0]["as_of_date"], "2026-07-28")
        report.pop("as_of_date")
        report.pop("data_sources")
        missing = self.market_result([report])["reports"][0]
        self.assertIsNone(missing["as_of_date"])
        self.assertEqual(missing["data_sources"], [])
        self.assertEqual(self.market_result([])["analysis_status"], "no_data")
        with patch.object(market, "MarketAgent", side_effect=FileNotFoundError("missing cache")):
            failure = json.loads(analyze_market_technicals())
        self.assertEqual(failure["analysis_status"], "error")
        self.assertIn("missing cache", failure["error"])

    def test_macro_retrieval_distinguishes_empty_failure_and_partial(self):
        repo = SimpleNamespace(get_macro_ts_features=Mock())
        vector = SimpleNamespace(query_vector_db=Mock())
        module = load_economy_file("nodes/data_retrieval.py", {
            "macro_agent.db.timescale.repository": repo,
            "macro_agent.db.vector.repository": vector,
        })
        docs = [{"content": "가상 기사", "metadata": {"source_type": "news"}}]
        metrics = {"test.rate": {"source": "synthetic", "as_of": "2026-07-28"}}
        cases = [(docs, metrics, False, "available"), ([], {}, False, "empty"),
                 (docs, {}, False, "partial"), ([], {}, True, "error"),
                 ([], metrics, True, "partial")]
        for documents, indicators, fail, expected in cases:
            with self.subTest(expected=expected, fail=fail):
                vector.query_vector_db.side_effect = RuntimeError("offline query failure") if fail else None
                vector.query_vector_db.return_value = documents
                repo.get_macro_ts_features.return_value = indicators
                state = module.data_retrieval_node({"question": "가상 경제 질문"})
                self.assertEqual(state["data_status"], expected)
                self.assertEqual(bool(state["warnings"]), fail)
        vector.query_vector_db.side_effect = None
        vector.query_vector_db.return_value = []
        repo.get_macro_ts_features.side_effect = RuntimeError("database unavailable")
        state = module.data_retrieval_node({"question": "가상 경제 질문"})
        self.assertEqual(state["data_status"], "error")
        self.assertTrue(state["warnings"])

    def test_macro_metadata_survives_public_agent_and_tool(self):
        forbidden = Mock(side_effect=AssertionError("DB/model initialization forbidden"))
        module = load_economy_file("agent.py", {
            "macro_agent.db.timescale.client": SimpleNamespace(initialize_schema=forbidden),
            "macro_agent.db.vector.repository": SimpleNamespace(ChromaReportStore=forbidden),
            "macro_agent.graph": SimpleNamespace(build_graph=forbidden),
            "macro_agent.tools.news_tool": SimpleNamespace(set_report_store=forbidden),
        })
        agent = object.__new__(module.MacroAnalysisAgent)
        document = {"source_name": "가상 자료", "url": "https://example.invalid/source",
                    "published_date": "2026-07-28"}
        state = {"is_domain": True, "data_status": "partial", "run_at": "2026-09-30T00:00:00Z",
                 "rag_docs": [{"metadata": document}], "ts_features": {},
                 "warnings": ["가상 시계열 조회 실패"],
                 "analysis_result": {"is_my_domain": True, "overall_risk_level": "LOW",
                                     "confidence_score": 0.2, "data_sources": ["가상 자료"]}}
        payload = self.macro_result(agent._extract_result(state))
        self.assertEqual(payload["analysis_status"], "partial")
        self.assertEqual(payload["evidence_metadata"]["documents"], [document | {"content": None}])
        self.assertEqual(payload["_agent_meta"]["metric_count"], 0)
        self.assertEqual(payload["_agent_meta"]["warnings"], state["warnings"])
        self.assertEqual(payload["overall_risk_level"], "LOW")  # A supported LOW is still allowed.
        state.update(rag_docs=[], ts_features={"test.rate": {"source": "test", "as_of": "2026-07-27"}})
        extracted = agent._extract_result(state)
        self.assertEqual(extracted["evidence_metadata"]["metrics"]["test.rate"]["as_of"], "2026-07-27")
        for status in ("empty", "error"):
            state.update(data_status=status, rag_docs=[], ts_features={})
            extracted = agent._extract_result(state)
            self.assertIsNone(extracted["overall_risk_level"])
            self.assertEqual(extracted["confidence_score"], 0.0)  # Public API float contract.
            payload = self.macro_result(extracted)
            self.assertIsNone(payload["overall_risk_level"])
            self.assertIsNone(payload["confidence_score"])
            self.assertTrue(payload["limitations"])
            self.assertEqual(payload["analysis_status"], "no_data" if status == "empty" else "error")
        forbidden.assert_not_called()

    def test_macro_rejection_parse_failure_and_exception(self):
        cases = [({"_agent_meta": {"is_domain": False}}, "out_of_domain"),
                 ({"_parse_error": True}, "error"),
                 ({"_agent_meta": {"fatal_error": "offline failure"}}, "error"),
                 ({"is_my_domain": False}, "unavailable")]
        for fields, expected in cases:
            with self.subTest(expected=expected):
                payload = self.macro_result({"overall_risk_level": "LOW", **fields})
                self.assertEqual(payload["analysis_status"], expected)
                self.assertIsNone(payload["overall_risk_level"])
        fake = SimpleNamespace(MacroAnalysisAgent=Mock(side_effect=RuntimeError("offline error")))
        with patch.dict(sys.modules, {"macro_agent.agent": fake}):
            self.assertEqual(json.loads(analyze_macro_economy("질문"))["analysis_status"], "error")

    def test_absent_risk_is_not_low_at_merge_source(self):
        module = load_economy_file("nodes/synthesizer.py", {})
        for a, b, expected in [(None, None, None), ("UNKNOWN", None, None),
                               (None, "LOW", "LOW"), ("HIGH", "MEDIUM", "HIGH")]:
            self.assertEqual(module._max_risk(a, b), expected)
        self.assertIsNone(module._merge_results({}, {}, "질문")["overall_risk_level"])

    def test_failed_tool_event_is_preserved(self):
        from autogen_agentchat.messages import ToolCallExecutionEvent
        from autogen_core.models import FunctionExecutionResult
        event = ToolCallExecutionEvent(source="EconomyExpert", content=[FunctionExecutionResult(
            name="analyze_macro_economy", call_id="failed-1", is_error=True, content="offline failure")])
        rendered = report_markdown("질문", [event], "2026-09-30T09:00:00+09:00")
        record = json.loads(rendered.split("```json\n")[1].split("\n```", 1)[0])
        self.assertTrue(record["is_error"])
        self.assertEqual(record["call_id"], "failed-1")

    def test_main_wires_tools_and_saves_original_evidence_without_model(self):
        from autogen_agentchat.messages import ToolCallExecutionEvent
        from autogen_core.models import FunctionExecutionResult
        from integration_tools import EVIDENCE_RULES
        event = ToolCallExecutionEvent(source="MarketExpert", content=[FunctionExecutionResult(
            name="analyze_market_technicals", call_id="wiring", is_error=False,
            content='{"as_of_date":"2026-07-28","data_sources":["synthetic"]}')])
        assistant = Mock()
        console = AsyncMock(return_value=SimpleNamespace(messages=[event]))
        # Prevent key-file reads, real client creation and actual team execution.
        with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test-placeholder"}), \
                patch("dotenv.load_dotenv"), \
                patch("autogen_ext.models.openai.OpenAIChatCompletionClient"), \
                patch("autogen_agentchat.agents.AssistantAgent", assistant), \
                patch("autogen_agentchat.teams.SelectorGroupChat"), \
                patch("autogen_agentchat.ui.Console", console):
            spec = importlib.util.spec_from_file_location("offline_main", ROOT / "main.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        definitions = {call.kwargs["name"]: call.kwargs for call in assistant.call_args_list}
        self.assertIs(definitions["MarketExpert"]["tools"][0].__wrapped__, analyze_market_technicals)
        self.assertIs(definitions["EconomyExpert"]["tools"][0].__wrapped__, analyze_macro_economy)
        self.assertIs(definitions["SearchExpert"]["tools"][0].__wrapped__, search_news_and_disclosures)
        self.assertIs(definitions["SemiExpert"]["tools"][0].__wrapped__, analyze_semiconductor_tech)
        for name in ("SearchExpert", "SemiExpert", "MarketExpert", "EconomyExpert", "Integrator"):
            self.assertIn(EVIDENCE_RULES, definitions[name]["system_message"])
        with tempfile.TemporaryDirectory() as tmp:
            module.BASE_DIR = tmp
            self.loop.run_until_complete(module.main("offline wiring check"))
            files = list((Path(tmp) / "reports").glob("*.md"))
            self.assertEqual(len(files), 1)
            stored = files[0].read_text(encoding="utf-8")
            self.assertIn("2026-07-28", stored)
            self.assertIn("synthetic", stored)


if __name__ == "__main__":
    unittest.main()
