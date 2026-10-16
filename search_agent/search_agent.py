"""
search_agent.py
반도체 관련 데이터 수집 에이전트.

다른 에이전트에서 호출:
    from search_agent import run
    result = run("삼성전자 HBM")

반환값:
    {
        "query": str,
        "news": [...],        # 네이버 뉴스 기사 리스트
        "disclosures": [...], # DART 공시 리스트
        "threads_posts": [...] # DB에 저장된 Threads 포스트 리스트
    }
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from naver_news_api import fetch_naver_news
from dart_api import fetch_corp_code, fetch_disclosures
from database import SessionLocal, init_db, save_news_to_db, save_disclosures_to_db
from models import ThreadsPost, ThreadsAccount


def _search_threads_from_db(query: str) -> list[dict]:
    """DB에 저장된 Threads 포스트 중 query 키워드를 포함하는 것을 반환."""
    db = SessionLocal()
    try:
        posts = db.query(ThreadsPost).join(ThreadsAccount).all()
        results = []
        for post in posts:
            if query.lower() in (post.content or "").lower():
                results.append({
                    "post_id": post.post_id,
                    "content": post.content,
                    "like_count": post.like_count,
                    "reply_count": post.reply_count,
                    "repost_count": post.repost_count,
                    "posted_at": post.posted_at,
                    "username": post.account.username if post.account else "",
                })
        return results
    finally:
        db.close()


def _build_payload(query: str, news: list, disclosures: list, threads_posts: list,
                   collection_status: dict | None = None) -> dict:
    """수집 결과를 integration_payload 형식으로 변환."""
    key_evidence = []
    for n in news:
        key_evidence.append(f"[뉴스] {n['title']} — {n['press']} ({n['pub_date']})")
    for d in disclosures:
        key_evidence.append(f"[공시] {d['title']} — {d['corp_name']} ({d['filed_at']})")
    for p in threads_posts:
        key_evidence.append(f"[SNS] @{p['username']}: {p['content'][:60]}...")

    limitations = []
    if not news:
        limitations.append("네이버 뉴스 수집 결과 없음")
    if not disclosures:
        limitations.append(f"'{query}' 관련 공시를 확보하지 못함. 미발견은 공시·계약 부재의 증거가 아닙니다.")

    documents = []
    for kind, rows in (("news", news), ("disclosure", disclosures), ("threads", threads_posts)):
        for i, row in enumerate(rows, 1):
            documents.append({
                "id": f"{kind}-{i}", "source_type": kind,
                "source_id": row.get("rcept_no") or row.get("post_id"),
                "title": row.get("title"), "source": row.get("press") or row.get("corp_name") or row.get("username"),
                "url": row.get("original_url") or row.get("url"),
                "published_at": row.get("published_at") or row.get("pub_date") or row.get("filed_at") or row.get("posted_at") or None,
                "data_as_of": None, "content": row.get("description") or row.get("content"),
            })
    statuses = collection_status or {"news": "unknown", "disclosures": "unknown", "threads": "unknown"}
    if all(s in {"available", "no_data"} for s in statuses.values()):
        status = "available" if documents else "no_data"
    else:
        status = "partial" if documents else "error" if "error" in statuses.values() else "unavailable"
    if status in {"partial", "error", "unavailable"}:
        limitations.append("일부 수집 경로가 미확인·실패·미실행 상태입니다. collection_status를 확인하세요.")

    total = len(news) + len(disclosures) + len(threads_posts)
    handoff_message = (
        f"'{query}' 관련 데이터 수집 완료. "
        f"뉴스 {len(news)}건 / 공시 {len(disclosures)}건 / SNS {len(threads_posts)}건 (총 {total}건)."
    )

    return {
        "agent_name": "Search Agent",
        "analysis_status": status,
        "collection_status": statuses,
        "evidence_metadata": {"documents": documents},
        "keyword": query,
        "news_count": len(news),
        "disclosure_count": len(disclosures),
        "threads_count": len(threads_posts),
        "key_evidence": key_evidence,
        "limitations": limitations,
        "handoff_message": handoff_message,
        "raw": {
            "news": news,
            "disclosures": disclosures,
            "threads_posts": threads_posts,
        }
    }


def run(query: str, news_count: int = 10, disclosure_count: int = 20) -> dict:
    """
    search_agent 진입점. 다른 에이전트에서 이 함수를 호출합니다.

    Args:
        query (str): 검색 키워드 (예: '삼성전자', 'HBM', 'SK하이닉스')
        news_count (int): 수집할 뉴스 기사 수 (기본 10)
        disclosure_count (int): 수집할 공시 수 (기본 20)

    Returns:
        dict: integration_payload 형식
              {'agent_name', 'keyword', 'news_count', 'disclosure_count',
               'threads_count', 'key_evidence', 'limitations', 'handoff_message', 'raw'}
    """
    init_db()
    print(f"\n[SearchAgent] 검색 시작: '{query}'")

    # 1. 네이버 뉴스 수집
    print("[SearchAgent] 네이버 뉴스 수집 중...")
    statuses = {}
    try:
        news = fetch_naver_news(query, display=news_count)
        statuses["news"] = "available" if news else "no_data"
    except Exception:
        news, statuses["news"] = [], "error"
    if news:
        save_news_to_db(news)
    print(f"[SearchAgent] 뉴스 {len(news)}건 수집 완료")

    # 2. DART 공시 수집
    print("[SearchAgent] DART 공시 수집 중...")
    disclosures = []
    try:
        corp = fetch_corp_code(query)
        if corp:
            disclosures = fetch_disclosures(corp_code=corp["corp_code"], display=disclosure_count)
            for d in disclosures:
                d["stock_code"] = corp.get("stock_code", "")
            statuses["disclosures"] = "available" if disclosures else "no_data"
        else:
            statuses["disclosures"] = "company_not_matched"
    except Exception:
        statuses["disclosures"] = "error"
    if disclosures:
        save_disclosures_to_db(disclosures)
    print(f"[SearchAgent] 공시 {len(disclosures)}건 수집 완료")

    # 3. Threads DB 조회
    print("[SearchAgent] Threads 포스트 DB 조회 중...")
    try:
        threads_posts = _search_threads_from_db(query)
        statuses["threads"] = "available" if threads_posts else "no_data"
    except Exception:
        threads_posts, statuses["threads"] = [], "error"
    print(f"[SearchAgent] Threads 포스트 {len(threads_posts)}건 조회 완료")

    payload = _build_payload(query, news, disclosures, threads_posts, statuses)

    print(f"[SearchAgent] 완료 - 뉴스 {len(news)}건 / 공시 {len(disclosures)}건 / Threads {len(threads_posts)}건\n")
    return payload


if __name__ == "__main__":
    import json
    keyword = sys.argv[1] if len(sys.argv) > 1 else "삼성전자"
    result = run(keyword)
    print(json.dumps(result, ensure_ascii=False, indent=2))
