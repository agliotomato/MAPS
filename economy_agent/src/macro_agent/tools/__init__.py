"""
macro_agent.tools — LangChain Tool 모음
LangGraph 노드에서 ALL_TOOLS를 주입하거나 개별 tool을 직접 임포트해 사용하세요.
"""

from macro_agent.tools.ecos_tool import fetch_ecos_data
from macro_agent.tools.fred_tool import fetch_fred_data
from macro_agent.tools.news_tool import (
    ReportStoreProtocol,
    scraped_news_and_reports,
    set_report_store,
    upload_report_to_rag,
)

# LangGraph 에이전트 노드에 바인딩할 전체 툴 목록
ALL_TOOLS = [
    fetch_ecos_data,
    fetch_fred_data,
    scraped_news_and_reports,
    upload_report_to_rag,
]

__all__ = [
    "fetch_ecos_data",
    "fetch_fred_data",
    "scraped_news_and_reports",
    "upload_report_to_rag",
    "set_report_store",
    "ReportStoreProtocol",
    "ALL_TOOLS",
]
