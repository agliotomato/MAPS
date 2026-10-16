"""Production collectors/adapters with network, DB and LLM boundaries replaced."""
import importlib.util
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "semiconductor_agent_design"))
from app.agent import models
import integration_tools as integration


def load_file(relative, replacements=None):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("offline_live_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module, **(replacements or {})}), patch("dotenv.load_dotenv"):
        spec.loader.exec_module(module)
    return module


class LiveEvidenceTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        network.start()
        self.addCleanup(network.stop)

    def test_disabled_qdrant_skips_embedding_and_preserves_search_filters(self):
        encode = Mock(side_effect=AssertionError("Embedding forbidden"))
        client = Mock(side_effect=AssertionError("Qdrant forbidden"))
        lexical = Mock(return_value=[
            {"doc_uid":"keep", "source":"wsts", "year":2025},
            {"doc_uid":"old", "source":"wsts", "year":2020},
            {"doc_uid":"other", "source":"other", "year":2025}])
        with patch.dict(os.environ, {"RAG_QDRANT_DISABLED":"1"}), patch.dict(sys.modules, {
                "app.rag.embedder":SimpleNamespace(encode_query=encode)}):
            module = load_file("semiconductor_agent_design/app/rag/retriever.py", {
                "app.db.qdrant":SimpleNamespace(get_qdrant_client=client,
                    DENSE_VECTOR_NAME="dense",SPARSE_VECTOR_NAME="sparse",SEMI_KNOWLEDGE_COLLECTION="semi_knowledge"),
                "app.rag.lexical":SimpleNamespace(search_lexical_payload=lexical)})
            hits = module.search("WSTS", filter_source="wsts", filter_year_min=2024)
        self.assertEqual([h["doc_uid"] for h in hits], ["keep"])
        encode.assert_not_called()
        client.assert_not_called()

    def test_search_real_adapter_preserves_sources_and_collection_states(self):
        news = Mock(return_value=[{"title": "테스트 뉴스", "press": "가상 언론",
            "pub_date": "2026-07-28 16:00:00", "published_at": "Tue, 28 Jul 2026 16:00:00 +0900",
            "original_url": "https://example.invalid/original", "url": "https://example.invalid/portal",
            "description": "개발용 근거"}])
        corp = Mock(return_value={"corp_code": "TEST"})
        disclosures = Mock(side_effect=RuntimeError("query failed"))
        db = Mock()
        db.query.return_value.join.return_value.all.return_value = []
        replacements = {
            "naver_news_api": SimpleNamespace(fetch_naver_news=news),
            "dart_api": SimpleNamespace(fetch_corp_code=corp, fetch_disclosures=disclosures),
            "database": SimpleNamespace(SessionLocal=Mock(return_value=db), init_db=Mock(),
                                        save_news_to_db=Mock(), save_disclosures_to_db=Mock()),
            "models": SimpleNamespace(ThreadsPost=object(), ThreadsAccount=object()),
        }
        with patch.dict(sys.modules, replacements):
            payload = json.loads(integration.search_news_and_disclosures("가상 기업"))
            self.assertEqual(payload["analysis_status"], "partial")
            self.assertEqual(payload["collection_status"]["disclosures"], "error")
            document = payload["evidence_metadata"]["documents"][0]
            self.assertEqual(document["url"], "https://example.invalid/original")
            self.assertTrue(document["published_at"].endswith("+0900"))
            self.assertIsNone(document["data_as_of"])
            table = integration.tool_metadata_markdown(json.dumps(payload))
            self.assertIn("partial", table)
            self.assertIn("| 미상 | https://example.invalid/original", table)
            news.return_value = []
            self.assertEqual(json.loads(integration.search_news_and_disclosures("질문"))["analysis_status"], "error")
            disclosures.side_effect = None
            disclosures.return_value = []
            self.assertEqual(json.loads(integration.search_news_and_disclosures("질문"))["analysis_status"], "no_data")
            corp.return_value = None
            self.assertEqual(json.loads(integration.search_news_and_disclosures("질문"))["analysis_status"], "unavailable")

    def test_collector_failure_is_not_an_empty_success_and_news_keeps_timezone(self):
        news = load_file("search_agent/naver_news_api.py")
        dart = load_file("search_agent/dart_api.py")
        news.NAVER_CLIENT_ID, news.NAVER_CLIENT_SECRET = "test", "test"
        dart.DART_API_KEY = "test"
        response = Mock()
        response.json.return_value = {"items": [{"title": "test", "pubDate": "Tue, 28 Jul 2026 16:00:00 +0900",
                                               "originallink": "https://example.invalid/original"}]}
        with patch.object(news.requests, "get", return_value=response) as request:
            item = news.fetch_naver_news("query")[0]
            self.assertEqual(item["published_at"], response.json.return_value["items"][0]["pubDate"])
            self.assertEqual(request.call_args.kwargs["timeout"], 15)
        with patch.object(news.requests, "get", side_effect=RuntimeError("SECRET-URL")):
            for call in (lambda: news.fetch_naver_news("q"), dart._load_corp_list, dart.fetch_disclosures):
                with self.assertRaises(RuntimeError) as raised:
                    call()
                self.assertNotIn("SECRET", str(raised.exception))
        response.json.return_value = {"status": "013"}
        with patch.object(dart.requests, "get", return_value=response):
            self.assertEqual(dart.fetch_disclosures(), [])
            response.json.return_value = {"status": "010", "message": "failure"}
            with self.assertRaises(RuntimeError):
                dart.fetch_disclosures()
            response.json.return_value = {"status": "000", "list": [
                {"rcept_no": "20261002000001", "rcept_dt": "20261002"}]}
            disclosure = dart.fetch_disclosures()[0]
            self.assertEqual(disclosure["url"],
                             "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20261002000001")

    def test_index_and_both_lexical_paths_preserve_publication_and_url(self):
        published = datetime(2026, 7, 28, 7, tzinfo=timezone.utc)
        row = dict(doc_uid="doc-test", chunk_index=0, chunk_text="HBM packaging research", source="test",
                   source_type="paper", title="HBM research", published_at=published,
                   url="https://example.invalid/paper", tags=[], extra={})
        postgres = SimpleNamespace(get_pg_conn=Mock())
        indexer = load_file("semiconductor_agent_design/app/rag/indexer.py", {
            "app.db.postgres": postgres,
            "app.db.qdrant": SimpleNamespace(ensure_semi_knowledge=Mock(), upsert_chunks_batch=Mock()),
            "app.rag.embedder": SimpleNamespace(EmbedderUnavailableError=RuntimeError, encode_batch=Mock()),
        })
        payload = indexer._build_payload(row)
        self.assertEqual(payload["published_at"], published.isoformat())
        self.assertEqual(payload["url"], row["url"])
        self.assertIsNone(indexer._build_payload({**row, "published_at": None})["published_at"])
        lexical = load_file("semiconductor_agent_design/app/rag/lexical.py")
        from unittest.mock import MagicMock
        connection = MagicMock()
        cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.description = [(key,) for key in row]
        cursor.fetchall.return_value = [tuple(row.values())]
        postgres.get_pg_conn.return_value = connection
        with patch.dict(sys.modules, {"app.db.postgres": postgres}):
            hits = lexical._search_postgres_chunks("HBM packaging", top_k=10)
        self.assertEqual(cursor.execute.call_count, 2)
        self.assertEqual(len(hits), 2)  # Exercise both the chunk and document query payloads.
        for hit in hits:
            self.assertEqual(hit["url"], row["url"])
            self.assertEqual(hit["published_at"], published.isoformat())
        for call in cursor.execute.call_args_list:
            self.assertIn("d.url", call.args[0])

    def test_rag_selection_does_not_invent_date_and_adapter_preserves_context(self):
        steps = SimpleNamespace(**{name: Mock() for name in (
            "calc_expected_value", "competitive_map", "context_collect", "market_check", "tech_evaluate")})
        pipeline = load_file("semiconductor_agent_design/app/agent/pipeline.py", {
            "app.agent.steps": steps, "app.agent.reporter": SimpleNamespace(build_report=Mock())})
        search = Mock(return_value=[{"title": "test", "text": "HBM", "year": 2024,
                                    "doc_uid": "doc-test", "url": "https://example.invalid/doc"}])
        with patch.dict(sys.modules, {"app.rag.retriever": SimpleNamespace(search=search)}), \
                patch.object(pipeline, "run", side_effect=lambda event, **kw: event):
            event = pipeline.run_from_rag("test", verbose=False)
        self.assertEqual(event.published_at, "")
        self.assertEqual(event.year, 2024)
        self.assertEqual(event.source_id, "doc-test")
        report = models.IntelligenceReport(event, models.TechContext(event, related_chunks=[{
            "doc_uid": "context-doc", "published_at": "2026-07-27", "url": "https://example.invalid/context"}]),
            models.TechEvaluation(3, 4, "unknown", False), models.CompetitiveMap(),
            models.ExpectedValue(0.5, 0.5, "medium", "mid", "high"),
            models.MarketCheck(None, None, "unavailable", analysis_status="no_data"),
            as_of="2026-10-04T00:00:00Z", full_report="X" * 3001 + "TAIL")
        with patch.dict(sys.modules, {"app.agent.pipeline": SimpleNamespace(run_from_rag=Mock(return_value=report))}):
            payload = json.loads(integration.analyze_semiconductor_tech("test"))
        self.assertEqual(payload["analysis_status"], "partial")
        self.assertIsNone(payload["evidence_metadata"]["documents"][0]["published_at"])
        self.assertEqual(payload["evidence_metadata"]["documents"][1]["source_id"], "context-doc")
        self.assertTrue(payload["full_report"].endswith("TAIL"))
        self.assertIsNone(payload["market"]["price_change_pct"])
        table = integration.tool_metadata_markdown(json.dumps(payload))
        self.assertNotIn(report.as_of, table)
        self.assertIn("반도체 내부 시장 조회: no_data", table)

    def test_semiconductor_missing_prices_never_become_zero_or_buy_signal(self):
        steps = load_file("semiconductor_agent_design/app/agent/steps.py", {
            "app.agent.llm": SimpleNamespace(call_llm_json=Mock()),
            "app.rag.evidence_builder": SimpleNamespace(build_evidence_pack=Mock())})
        reporter = load_file("semiconductor_agent_design/app/agent/reporter.py", {
            "app.agent.llm": SimpleNamespace(call_llm=Mock())})
        event = models.TechEvent("test", "", "paper", "test", "")
        ev = models.ExpectedValue(0.8, 0.8, "high", "mid", "high")
        competitive = models.CompetitiveMap(beneficiaries=[{"code": "TEST", "company": "test", "reason": "test"}])
        evaluation = models.TechEvaluation(3, 4, "unknown", False)
        for result, error, status in [([], None, "no_data"), ([], RuntimeError("failed"), "error")]:
            with patch.object(steps, "_get_company_price_changes", return_value=result, side_effect=error):
                market = steps.market_check(event, ev, competitive)
            self.assertEqual(market.analysis_status, status)
            self.assertIsNone(market.price_change_pct)
            self.assertIsNone(market.already_priced_in)
            self.assertEqual(market.signal, "unavailable")
            rendered = reporter._render_markdown(event, evaluation, competitive, ev, market)
            self.assertIn("판단 유보", rendered)
            self.assertNotIn("+0.0%", rendered)
            self.assertIn("게시일: 미상", rendered)
        move = dict(code="TEST", company="test", change_pct=0.0,
                    observed_from="2026-07-27", data_as_of="2026-07-28", source="price_daily")
        with patch.object(steps, "_get_company_price_changes", return_value=[move]) as fetch:
            market = steps.market_check(event, ev, competitive)
        fetch.assert_called_once()
        self.assertEqual(market.analysis_status, "available")
        self.assertEqual(market.price_change_pct, 0.0)  # Actual zero remains a measured value.
        self.assertEqual(market.company_moves[0]["data_as_of"], "2026-07-28")
        competitive.beneficiaries.append({"code": "MISSING", "company": "missing", "reason": "test"})
        with patch.object(steps, "_get_company_price_changes", return_value=[move]):
            partial = steps.market_check(event, ev, competitive)
        self.assertEqual(partial.analysis_status, "partial")
        self.assertIn("MISSING", partial.limitations[0])
        from unittest.mock import MagicMock
        connection = MagicMock()
        cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [("TEST", 0, date(2026, 7, 27), date(2026, 7, 28))]
        postgres = SimpleNamespace(get_pg_conn=Mock(return_value=connection))
        with patch.dict(sys.modules, {"app.db.postgres": postgres}):
            observations = steps._get_company_price_changes(["TEST"])
            self.assertEqual(observations[0]["data_as_of"], "2026-07-28")
            self.assertIn("latest.trade_date > oldest.trade_date", cursor.execute.call_args.args[0])
            self.assertIn("trade_date <= CURRENT_DATE", cursor.execute.call_args.args[0])
            postgres.get_pg_conn.side_effect = RuntimeError("DB failed")
            with self.assertRaises(RuntimeError):
                steps._get_company_price_changes(["TEST"])


    def test_undated_wsts_directory_does_not_use_collection_time(self):
        from unittest.mock import MagicMock
        connection = MagicMock()
        cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
        collector = load_file("semiconductor_agent_design/app/collectors/industry_collector.py", {
            "app.db.postgres": SimpleNamespace(get_pg_conn=Mock(return_value=connection))})
        now = datetime(2026, 10, 5, tzinfo=timezone.utc)
        with patch.dict(sys.modules, {"feedparser": SimpleNamespace()}), \
                patch.object(collector, "WSTS_COVERAGE_RSS", []), \
                patch.object(collector, "_now_utc", return_value=now):
            collector.collect_wsts_bluebook()
        cursor.execute.assert_called_once()
        sql, values = cursor.execute.call_args.args
        self.assertEqual(values[1], "wsts")
        self.assertIsNone(values[5])  # Publication date is unknown.
        self.assertEqual(values[6], now)  # Collection time stays separate.
        # Check upsert semantics locally; the production SQL also uses JSONB
        # for unrelated metadata, so adapt only placeholders and that cast.
        import sqlite3
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE tech_documents (doc_uid TEXT PRIMARY KEY, source, "
                       "source_type, title, url, published_at, collected_at, summary, "
                       "content, tags, confidence, extra)")
            query = sql.replace("%s", "?").replace("::jsonb", "")
            params = [str(v) if isinstance(v, (list, datetime)) else v for v in values]
            params[5] = "2026-05-08"
            db.execute(query, params)
            params[5] = None
            db.execute(query, params)
            self.assertIsNone(db.execute("SELECT published_at FROM tech_documents").fetchone()[0])
            params[0], params[2], params[5] = "dated-article", "wsts_article", "2026-05-07"
            db.execute(query, params)
            params[5] = None
            db.execute(query, params)
            self.assertEqual(db.execute("SELECT published_at FROM tech_documents "
                                        "WHERE doc_uid='dated-article'").fetchone()[0], "2026-05-07")


if __name__ == "__main__":
    unittest.main()
