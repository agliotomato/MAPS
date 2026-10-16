"""Exercise production adapters -> actual team/main -> checked saved report offline."""
import asyncio
from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from evidence_contract import planned_next_checks
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_live_evidence import load_file, models
from test_integration_evidence import load_economy_file
import integration_tools as integration
from evidence_contract import (ProductionReport, SourceScope, check_report, source_scope, bind_report_type,
                               scope_answer, scope_description, review_policy, review_explanation)
from evidence_claims import claim_for_quote
from agent_team import TOOL_NAMES, EXPERT_NAMES
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import StructuredMessage, ToolCallExecutionEvent
from autogen_core import FunctionCall
from autogen_core.models import CreateResult, RequestUsage, FunctionExecutionResult
from autogen_ext.models.replay import ReplayChatCompletionClient


def production_fixtures():
    """Real adapters, synthetic upstream collectors/DB results; no external calls."""
    news = {"title": "가상 A사와 B사", "press": "synthetic", "published_at": "2026-10-12",
            "original_url": "https://example.invalid/news", "url": "https://example.invalid/news",
            "pub_date": "2026-10-12",
            "description": "가상 A사와 B사의 태그가 함께 붙은 글. 회사별 주장 연결은 미확인."}
    db = Mock()
    db.query.return_value.join.return_value.all.return_value = []
    with patch.dict(sys.modules, {
        "naver_news_api": SimpleNamespace(fetch_naver_news=Mock(return_value=[news])),
        "dart_api": SimpleNamespace(fetch_corp_code=Mock(return_value={"corp_code": "TEST"}),
                                    fetch_disclosures=Mock(return_value=[])),
        "database": SimpleNamespace(SessionLocal=Mock(return_value=db), init_db=Mock(),
                                    save_news_to_db=Mock(), save_disclosures_to_db=Mock()),
        "models": SimpleNamespace(ThreadsPost=object(), ThreadsAccount=object()),
    }):
        search = integration.search_news_and_disclosures("가상 A사")

    forbidden = Mock(side_effect=AssertionError("DB/LLM forbidden"))
    macro = load_economy_file("agent.py", {
        "macro_agent.db.timescale.client": SimpleNamespace(initialize_schema=forbidden),
        "macro_agent.db.vector.repository": SimpleNamespace(ChromaReportStore=forbidden),
        "macro_agent.graph": SimpleNamespace(build_graph=forbidden),
        "macro_agent.tools.news_tool": SimpleNamespace(set_report_store=forbidden)})
    state = {"is_domain": True, "data_status": "partial", "ts_features": {},
        "rag_docs": [{"content": "가상 A사의 배당 매력이 높다는 투자 해설. 매출·수요 관측치는 제공되지 않음.",
                      "metadata": {"url": "https://example.invalid/opinion", "published_at": "2026-10-12"}}],
        "analysis_result": {"summary": "배당 관련 투자 해설", "is_my_domain": True}}
    upstream = object.__new__(macro.MacroAnalysisAgent)._extract_result(state)
    cls = Mock()
    cls.return_value.invoke.return_value = upstream
    with patch.dict(sys.modules, {"macro_agent.agent": SimpleNamespace(MacroAnalysisAgent=cls)}):
        economy = integration.analyze_macro_economy("가상 A사")

    import pandas as pd
    from market_agent import market_agent as market
    prices = pd.DataFrame({"Date": pd.date_range("2026-10-04", periods=9), "AssetCode": "TEST",
        "AssetName": "가상 A사", "Market": "TEST", "Close": range(100, 109), "Source": "synthetic"})
    report = market.MarketReporter().generate_report_for_asset(prices.iloc[-1])
    with patch.object(market.MarketAgent, "run", return_value={"price_df": prices,
            "report": {"json": {"generated_at": "2026-10-13", "reports": [report]}}}):
        market_payload = integration.analyze_market_technicals()

    event = models.TechEvent("가상 제품 심사", "배경 정보. " * 310 +
        "가상 제품은 승인될 가능성이 있으나 아직 승인 확정은 아님.", "news", "synthetic",
        "2026-10-12", source_id="TEST-EVENT")
    ctx = models.TechContext(event, related_chunks=[{"text": "다른 제품의 승인이 보도되었으나 직접 확인 자료는 없음.",
        "doc_uid": "TEST-CONTEXT", "published_at": "2026-10-12"}],
        context_text="다른 제품의 승인이 보도되었으나 직접 확인 자료는 없음.")
    llm = Mock(return_value={"innovation_score": 3, "trl": 4, "reproducibility": "low",
                            "prior_art_exists": False, "summary": "승인 상태 미확인"})
    steps = load_file("semiconductor_agent_design/app/agent/steps.py", {
        "app.agent.llm": SimpleNamespace(call_llm_json=llm),
        "app.rag.evidence_builder": SimpleNamespace(build_evidence_pack=forbidden)})
    evaluation = steps.tech_evaluate(ctx)
    assert "아직 승인 확정은 아님" in llm.call_args.args[1]
    report = models.IntelligenceReport(event, ctx, evaluation, models.CompetitiveMap(),
        models.ExpectedValue(0.5, 0.5, "medium", "mid", "medium"),
        models.MarketCheck(None, None, "unavailable", analysis_status="no_data"),
        headline="가상 제품 분석", full_report="가상 제품 승인이 완료됐습니다.")
    with patch.dict(sys.modules, {"app.agent.pipeline": SimpleNamespace(run_from_rag=Mock(return_value=report))}):
        technology = integration.analyze_semiconductor_tech("가상 제품")
    return dict(zip(TOOL_NAMES, [search, economy, market_payload, technology]))


def tool_messages(fixtures):
    return [ToolCallExecutionEvent(source=owner, content=[FunctionExecutionResult(
        name=name, call_id=name, is_error=False, content=fixtures[name])])
        for name, owner in zip(TOOL_NAMES, EXPERT_NAMES)]


def valid_final(fixtures):
    reviews, sources = [], []
    for payload in fixtures.values():
        for source in json.loads(payload)["evidence_handoff"]["sources"]:
            sources.append(source)
            quote = source["text"][:50]
            material, status = review_policy(source, quote)
            reviews.append(dict(source_id=source["id"], source_scope=source["source_scope"],
                quote=quote, material_type=material, claim_status=status, claim=claim_for_quote(source, quote),
                company_link="uncertain", interpretation=review_explanation(source, quote)))
    return ProductionReport(answer=scope_answer(sources), price_evidence="present",
        source_reviews=reviews, judgment="판단 유보", confidence="Low", outlook_7d="판단 유보",
        outlook_14d="판단 유보", outlook_30d="판단 유보", next_checks=planned_next_checks(sources))


def replay_client(final):
    responses = [CreateResult(finish_reason="function_calls", cached=False,
        usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
        content=[FunctionCall(id=name, name=name, arguments=json.dumps(args))])
        for name, args in zip(TOOL_NAMES, ({"query": "가상 A사"}, {"question": "가상 A사"},
                                          {"collect_data": False}, {"query": "가상 제품"}))]
    responses.append(final.model_dump_json())
    return ReplayChatCompletionClient(responses, model_info={"vision": False, "function_calling": True,
        "json_output": True, "structured_output": True, "family": "unknown"})


async def run_production(destination, fixtures, client, task=None, independent=False):
    """Run actual main/build_team/report writing, replace only tool upstream and client factory."""
    calls, completed = [], []
    def deliver(name):
        calls.append(name)
        return fixtures[name]
    def search_news_and_disclosures(query: str, news_count: int = 10) -> str:
        """Retrieve frozen synthetic news."""
        return deliver(TOOL_NAMES[0])
    def analyze_macro_economy(question: str) -> str:
        """Retrieve frozen synthetic economic evidence."""
        return deliver(TOOL_NAMES[1])
    def analyze_market_technicals(collect_data: bool = False) -> str:
        """Retrieve frozen synthetic price evidence."""
        return deliver(TOOL_NAMES[2])
    def analyze_semiconductor_tech(query: str) -> str:
        """Retrieve frozen synthetic technical evidence."""
        return deliver(TOOL_NAMES[3])
    async def collect(stream):
        result = None
        async for item in stream:
            if isinstance(item, TaskResult):
                completed.append(item)
                result = item
        if result is None:
            raise AssertionError("No TaskResult")
        return result
    with ExitStack() as stack:
        if not independent:
            from agent_team import build_team as original_build
            def legacy_build(*args, **kwargs):
                return original_build(*args, **dict(kwargs, independent_reasoning=False))
            stack.enter_context(patch('agent_team.build_team', legacy_build))
        stack.enter_context(patch.dict("os.environ", {"OPENAI_API_KEY": "factory-replaced"}))
        stack.enter_context(patch("dotenv.load_dotenv"))
        stack.enter_context(patch("autogen_ext.models.openai.OpenAIChatCompletionClient", return_value=client))
        stack.enter_context(patch("autogen_agentchat.ui.Console", collect))
        for function in (search_news_and_disclosures, analyze_macro_economy,
                         analyze_market_technicals, analyze_semiconductor_tech):
            stack.enter_context(patch.object(integration, function.__name__, function))
        spec = importlib.util.spec_from_file_location("checked_production_main", ROOT / "main.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        entry.BASE_DIR = str(destination)
        await entry.main(task or "가상 A사의 2026-10-12 기준 근거·가격·승인 상태와 7/14/30일 전망을 분석해줘. 합성 검증 자료입니다.")
    return completed[0], calls


class ProductionReportTests(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        self.addCleanup(self.loop.close)
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        network.start()
        self.addCleanup(network.stop)
        self.fixtures = production_fixtures()

    def test_original_and_omission_survive_real_adapters(self):
        payloads = {k: json.loads(v) for k, v in self.fixtures.items()}
        econ = payloads[TOOL_NAMES[1]]["evidence_handoff"]["sources"][0]
        self.assertIn("투자 해설", econ["text"])
        prices = payloads[TOOL_NAMES[2]]["evidence_handoff"]["sources"][0]
        self.assertEqual(prices["source_scope"], source_scope(9, 9, "rows"))
        tech = payloads[TOOL_NAMES[3]]
        self.assertIn("아직 승인 확정은 아님", tech["evidence_handoff"]["sources"][0]["text"])
        self.assertEqual(tech["analysis_inputs"]["tech_evaluate"]["event"]["state"], "complete")
        self.assertEqual(tech["evidence_handoff"]["sources"][-1]["source_scope"]["state"], "absent")

    def test_clipped_technology_prompt_records_actual_counts(self):
        llm = Mock(return_value={})
        steps = load_file("semiconductor_agent_design/app/agent/steps.py", {
            "app.agent.llm": SimpleNamespace(call_llm_json=llm),
            "app.rag.evidence_builder": SimpleNamespace(build_evidence_pack=Mock())})
        ctx = models.TechContext(models.TechEvent("t", "X" * 12005, "news", "s", ""),
                                 context_text="Y" * 8010)
        steps.tech_evaluate(ctx)
        self.assertEqual(ctx.analysis_inputs["tech_evaluate"]["event"], source_scope(12005, 12000))
        self.assertEqual(ctx.analysis_inputs["tech_evaluate"]["context"], source_scope(8010, 8000))
        self.assertIn('"omitted": 10', llm.call_args.args[1])

    def test_macro_clipping_is_explicit_and_valid_json(self):
        prompt = load_economy_file("prompts/macro_prompt.py", {})
        value = {"rate": "X" * 9000}
        packed = json.loads(prompt.format_tool_data(value))
        self.assertEqual(packed["source_scope"], source_scope(
            len(json.dumps(value, ensure_ascii=False, indent=2)), 8000))
        self.assertEqual(len(packed["text"]), 8000)

    def test_retrieved_context_tail_is_preserved_before_stage_scope(self):
        original = "배경. " * 300 + "승인 가능성일 뿐 확정은 아님."
        raw = dict(id="doc:0", doc_uid="doc", title="HBM", text=original,
                   source="samsung_global_newsroom", source_type="news",
                   published_at="2026-10-12", year=2026, score=1)
        search = Mock(return_value=[raw])
        builder = load_file("semiconductor_agent_design/app/rag/evidence_builder.py", {
            "app.rag.lexical": SimpleNamespace(search_multi_query_lexical=search),
            "app.rag.retriever": SimpleNamespace(search_multi_query=search),
            "app.taxonomy": SimpleNamespace(taxonomy_analysis=lambda: {})})
        pack = builder.build_evidence_pack("HBM", top_k=1)
        self.assertEqual(pack.items[0].text, original)
        llm = Mock(return_value={})
        steps = load_file("semiconductor_agent_design/app/agent/steps.py", {
            "app.agent.llm": SimpleNamespace(call_llm_json=llm),
            "app.rag.evidence_builder": SimpleNamespace(build_evidence_pack=lambda *a, **kw: pack)})
        ctx = steps.context_collect(models.TechEvent("HBM", "", "news", "s", ""))
        steps.tech_evaluate(ctx)
        self.assertIn("승인 가능성일 뿐 확정은 아님.", llm.call_args.args[1])
        self.assertEqual(ctx.analysis_inputs["tech_evaluate"]["context"],
                         source_scope(len(ctx.context_text), len(ctx.context_text)))

    def test_validator_rejects_missing_sources_false_absence_and_invented_quotes(self):
        events, final = tool_messages(self.fixtures), valid_final(self.fixtures)
        def check(value, receipts=events):
            return check_report(receipts + [StructuredMessage[ProductionReport](source="Integrator", content=value)])[0]
        self.assertEqual(check(final)["status"], "contract_passed")
        for field, value, expected in [
            ("price_evidence", "absent", "price_presence_mismatch"),
            ("source_reviews", final.source_reviews[:-1], "source_review_coverage_mismatch")]:
            with self.subTest(field=field):
                broken = final.model_copy(update={field: value})
                self.assertIn(expected, check(broken)["issues"])
        broken = copy.deepcopy(final)
        broken.source_reviews[0].quote = "원문에 없는 인용"
        self.assertTrue(any("quote_not_in" in s for s in check(broken)["issues"]))
        broken = copy.deepcopy(final)
        broken.source_reviews[2].source_scope = SourceScope(**source_scope(0, 0, "rows"))
        # Revalidate an edited JSON object, as an actual model reply would be parsed.
        broken = ProductionReport.model_validate_json(broken.model_dump_json())
        self.assertTrue(any("source_scope_mismatch" in s for s in check(broken)["issues"]))
        self.assertIn("missing_tool_receipts", check(final, events[:-1])["issues"])
        self.assertEqual(check(final, events + [events[0]])["status"], "contract_passed")
        altered = copy.deepcopy(self.fixtures)
        value = json.loads(altered[TOOL_NAMES[0]])
        value["evidence_handoff"]["sources"][0]["text"] = "X" * 39
        altered[TOOL_NAMES[0]] = json.dumps(value)
        self.assertEqual(check(final, events + [tool_messages(altered)[0]])["status"], "needs_review")
        # Matching structured counts must not let a contradictory free explanation pass.
        broken = copy.deepcopy(final)
        broken.answer = "가격 원본이 없습니다."
        self.assertIn("scope_answer_mismatch", check(broken)["issues"])
        broken = copy.deepcopy(final)
        broken.source_reviews[2].interpretation = "가격 관련 자료가 전혀 없음."
        self.assertTrue(any("scope_description_mismatch" in s for s in check(broken)["issues"]))

    def test_generation_schema_binds_all_receipts_and_price_presence(self):
        sources = [s for raw in self.fixtures.values() for s in json.loads(raw)["evidence_handoff"]["sources"]]
        bound = bind_report_type(lambda: sources)
        schema = bound.model_json_schema()
        reviews = schema["properties"]["source_reviews"]
        self.assertEqual(reviews["required"], [s["id"] for s in sources])
        self.assertEqual(schema["properties"]["price_evidence"]["enum"], ["present"])
        self.assertNotIn("\n", schema["properties"]["answer"]["enum"][0])
        for source in sources:
            properties = reviews["properties"][source["id"]]["properties"]
            self.assertEqual(properties["source_id"]["enum"], [source["id"]])
            for key, value in source["source_scope"].items():
                self.assertEqual(properties["source_scope"]["properties"][key]["enum"], [value])
        final = valid_final(self.fixtures).model_dump()
        final["source_reviews"] = {row["source_id"]: row for row in final["source_reviews"]}
        self.assertEqual(len(bound.model_validate(final).source_reviews), len(sources))

    def test_actual_main_saves_checked_report_and_keeps_failed_draft(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as folder:
                final = valid_final(self.fixtures)
                if invalid:
                    final.price_evidence = "absent"
                client = replay_client(final)
                result, calls = self.loop.run_until_complete(run_production(Path(folder), self.fixtures, client))
                saved = next((Path(folder) / "reports").glob("*.md")).read_text(encoding="utf-8")
                self.assertEqual(calls, list(TOOL_NAMES))
                self.assertEqual(len(client.create_calls), 5)
                self.assertIn("needs_review" if invalid else "contract_passed", saved)
                self.assertIn("모델 원본 — 미검수", saved)
                self.assertIn("가격 원본 존재 여부(코드 집계): **present**", saved)
                self.assertEqual(result.messages[-1].content.price_evidence, "absent" if invalid else "present")


if __name__ == "__main__":
    unittest.main()
