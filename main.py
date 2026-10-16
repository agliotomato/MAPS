"""
AutoGen 기반 반도체 주가 예측 멀티에이전트 시스템

4개 전문 에이전트(Search, Economy, Market, Semiconductor)를
AutoGen SelectorGroupChat으로 통합하여 대화형 주가 예측을 수행합니다.

사용법:
    python main.py
    python main.py "삼성전자 주가 전망을 분석해줘"

환경변수:
    OPENAI_API_KEY  : OpenAI API 키 (필수)
    기타 키는 .env 파일 참조

구조:
    User → SelectorGroupChat (성공한 도구 결과에 따라 고정 순서 선택)
            ├── SearchExpert     (뉴스/공시/SNS 수집)
            ├── EconomyExpert    (거시경제/지정학 리스크)
            ├── MarketExpert     (기술적 분석/수급/타이밍)
            ├── SemiExpert       (반도체 기술 이벤트)
            └── Integrator       (종합 판단 → 주가 전망)
"""

import asyncio
import os
import sys
from pathlib import Path
from agent_team import build_team
from evidence_contract import ProductionReport, PRODUCTION_REPORT_RULES
from integration_tools import (
    analyze_macro_economy, analyze_market_technicals, report_markdown,
    search_news_and_disclosures, analyze_semiconductor_tech,
)

# ── Windows 콘솔 UTF-8 인코딩 설정 (cp949 에러 방지) ──
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── 경로 설정 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "economy_agent", "src"))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "semiconductor_agent_design"))

# ── 통합 .env 로딩 (각 에이전트 import 전에 먼저 로드) ──
from dotenv import load_dotenv

_env_file = Path(BASE_DIR) / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=True)
    print(f"[ENV] 통합 .env 로드 완료")
else:
    print(f"[WARN] .env 파일 없음 — .env.example을 .env로 복사하고 키를 설정하세요.")

from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ═══════════════════════════════════════════════════════════════
# LLM 클라이언트
# ═══════════════════════════════════════════════════════════════

def _load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    key_file = os.path.join(BASE_DIR, "OpenAI_key.txt")
    if os.path.exists(key_file):
        raw = open(key_file, "r", encoding="utf-8").read().strip()
        if "=" in raw and "OPENAI_API_KEY" in raw:
            raw = raw.split("=", 1)[1].strip()
        key = raw.strip('"').strip("'")
        os.environ["OPENAI_API_KEY"] = key
        return key
    raise ValueError(
        "OPENAI_API_KEY를 찾을 수 없습니다.\n"
        "  .env 파일에 OPENAI_API_KEY=sk-... 추가하세요."
    )


API_KEY = _load_api_key()

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=API_KEY,
)


# ═══════════════════════════════════════════════════════════════
# AutoGen 에이전트 정의
# ═══════════════════════════════════════════════════════════════

team = build_team(model_client, {
    "search_news_and_disclosures": search_news_and_disclosures,
    "analyze_macro_economy": analyze_macro_economy,
    "analyze_market_technicals": analyze_market_technicals,
    "analyze_semiconductor_tech": analyze_semiconductor_tech,
}, final_output_type=ProductionReport, extra_instructions=PRODUCTION_REPORT_RULES,
   independent_reasoning=True, peer_review='--peer-review' in sys.argv)


# ═══════════════════════════════════════════════════════════════
# 실행
# ═══════════════════════════════════════════════════════════════

async def main(task: str):
    # A current request without any date uses the explicitly displayed Korean day.
    import re
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from evidence_contract import target_company_aliases
    if target_company_aliases(task) and not re.search(r'\d{4}|거래일', task):
        task += f" (분석 기준일: {datetime.now(ZoneInfo('Asia/Seoul')).date().isoformat()}, 전망 단위: 달력일)"
    print("=" * 60)
    print(" AutoGen 반도체 주가 예측 멀티에이전트 시스템")
    print("=" * 60)
    print()
    print(" 전문가 에이전트:")
    print("   - SearchExpert     : 뉴스/공시/SNS 수집")
    print("   - EconomyExpert    : 거시경제/지정학 리스크")
    print("   - MarketExpert     : 기술적 분석/수급/타이밍")
    print("   - SemiExpert       : 반도체 기술 이벤트")
    print("   - Integrator       : 종합 판단")
    print()
    print(f" [시작] '{task}'")
    print("=" * 60)
    print()

    result = await Console(team.run_stream(task=task))

    # ── 보고서 파일 저장 ──
    from datetime import datetime

    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(reports_dir, f"report_{timestamp}.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_markdown(task, result.messages, datetime.now().astimezone().isoformat(),
                                validate_final=True))

    print()
    print(f"[보고서 저장 완료] {report_path}")


if __name__ == "__main__":
    task_args = [arg for arg in sys.argv[1:] if arg != '--peer-review']
    if task_args:
        task = " ".join(task_args)
    else:
        task = input("\n분석할 내용을 입력하세요: ").strip()
        if not task:
            task = "삼성전자 주가 전망을 분석해줘"

    asyncio.run(main(task))
