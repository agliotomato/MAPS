"""Shared production roles and orchestration for live tools and stored-evidence cases."""
from integration_tools import EVIDENCE_RULES, FINAL_REPORT_RULES

TOOL_NAMES = (
    "search_news_and_disclosures", "analyze_macro_economy",
    "analyze_market_technicals", "analyze_semiconductor_tech",
)
EXPERT_NAMES = ("SearchExpert", "EconomyExpert", "MarketExpert", "SemiExpert")


def next_speaker(messages):
    """Advance only after an assigned tool actually succeeded, never on model claims."""
    completed = set()
    owners = dict(zip(TOOL_NAMES, EXPERT_NAMES))
    for message in messages:
        if message.type == "TextMessage" and message.source == "user":
            completed.clear()
        if message.type == "ToolCallExecutionEvent":
            for result in message.content:
                if not result.is_error and owners.get(result.name) == message.source:
                    completed.add(result.name)
    return next((role for tool, role in zip(TOOL_NAMES, EXPERT_NAMES) if tool not in completed), "Integrator")


def build_team(model_client, tool_functions, *, extra_instructions="", max_turns=None,
               final_output_type=None, independent_reasoning=False, peer_review=False):
    """Use the same roles with either live or stored tools; no client is created here."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import SelectorGroupChat
    from autogen_agentchat.conditions import SourceMatchTermination
    from autogen_agentchat.messages import StructuredMessage

    if set(tool_functions) != set(TOOL_NAMES):
        raise ValueError("Exactly the four production tools are required")
    if peer_review and not independent_reasoning:
        raise ValueError('Peer review requires independent reasoning')
    from evidence_contract import ProductionReport, bind_report_type
    selector = next_speaker
    production_output = final_output_type is ProductionReport
    if final_output_type is ProductionReport:
        import json
        from functools import wraps
        receipts = {}
        active_task = None
        active_task_text = ""
        def selector(messages):
            nonlocal active_task, active_task_text
            user_message = next((message for message in reversed(messages)
                                 if message.source == "user"), None)
            task = user_message.id if user_message else None
            if task != active_task:
                receipts.clear()
                active_task = task
                active_task_text = str(user_message.content) if user_message else ""
            return next_speaker(messages)
        def capture(function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                result = function(*args, **kwargs)
                try:
                    for source in json.loads(result)["evidence_handoff"]["sources"]:
                        receipts.setdefault(source["id"], source)
                except (ValueError, TypeError, KeyError):
                    pass  # Final validation rejects missing or conflicting receipts.
                return result
            return wrapped
        tool_functions = {name: capture(function) for name, function in tool_functions.items()}
        final_output_type = bind_report_type(lambda: list(receipts.values()), lambda: active_task_text)
    search_news_and_disclosures = tool_functions[TOOL_NAMES[0]]
    analyze_macro_economy = tool_functions[TOOL_NAMES[1]]
    analyze_market_technicals = tool_functions[TOOL_NAMES[2]]
    analyze_semiconductor_tech = tool_functions[TOOL_NAMES[3]]

    search_expert = AssistantAgent(
        name="SearchExpert",
        model_client=model_client,
        tools=[search_news_and_disclosures],
        system_message=(
            "당신은 반도체 뉴스/공시/SNS 데이터 수집 전문가입니다.\n"
            "search_news_and_disclosures 도구를 호출하여 관련 데이터를 수집하세요.\n"
            "수집한 결과의 핵심 뉴스와 공시를 간결하게 요약 보고하세요.\n"
            "보고 범위: 뉴스 헤드라인, 공시 제목, SNS 동향만 다룹니다.\n"
            "거시경제, 기술적 분석, 반도체 기술은 다른 전문가에게 맡기세요."
        ) + EVIDENCE_RULES + extra_instructions,
    )

    economy_expert = AssistantAgent(
        name="EconomyExpert",
        model_client=model_client,
        tools=[analyze_macro_economy],
        system_message=(
            "당신은 거시경제/지정학 리스크 분석 전문가입니다.\n"
            "반드시 analyze_macro_economy 도구를 호출하세요. 도구 호출 없이 답변하지 마세요.\n"
            "분석 범위: 금리, 환율, 무역정책, 지정학적 리스크, PMI, 수출 동향.\n"
            "도구 결과의 핵심 리스크 레벨과 시그널을 보고하세요.\n"
            "뉴스 수집, 기술적 분석, 반도체 기술은 다른 전문가에게 맡기세요."
        ) + EVIDENCE_RULES + extra_instructions,
    )

    market_expert = AssistantAgent(
        name="MarketExpert",
        model_client=model_client,
        tools=[analyze_market_technicals],
        system_message=(
            "당신은 시장 기술적 분석 전문가입니다.\n"
            "반드시 analyze_market_technicals 도구를 호출하세요. 도구 호출 없이 답변하지 마세요.\n"
            "Timing Readiness(0-100) 점수와 진입 적합 구간을 보고하세요.\n"
            "분석 범위: 가격, 거래량, 이동평균, RSI, MACD, 외국인/기관 수급.\n"
            "뉴스, 거시경제, 반도체 기술은 다른 전문가에게 맡기세요."
        ) + EVIDENCE_RULES + extra_instructions,
    )

    semi_expert = AssistantAgent(
        name="SemiExpert",
        model_client=model_client,
        tools=[analyze_semiconductor_tech],
        system_message=(
            "당신은 반도체 기술 분석 전문가입니다.\n"
            "반드시 analyze_semiconductor_tech 도구를 호출하세요. 도구 호출 없이 답변하지 마세요.\n"
            "기댓값(EV score), 시장 반영 여부, 수혜/위협 기업을 보고하세요.\n"
            "뉴스 수집, 거시경제, 시장 기술적 분석은 다른 전문가에게 맡기세요."
        ) + EVIDENCE_RULES + extra_instructions,
    )

    integrator = AssistantAgent(
        name="Integrator",
        model_client=model_client,
        output_content_type=final_output_type,
        system_message=(
            "당신은 반도체 주가 예측 통합 분석가입니다.\n"
            "다른 전문가들(SearchExpert, EconomyExpert, MarketExpert, SemiExpert)의\n"
            "도구 근거를 종합하여 사용자의 질문에 답하세요.\n\n"
            "중요: 4개 전문가 에이전트의 tool 호출 결과가 대화에 모두 나타난 후에만 종합 보고서를 작성하세요.\n"
            "아직 tool 호출을 하지 않은 에이전트가 있다면 '(에이전트이름)의 분석을 요청합니다'라고만 말하세요.\n\n"
            "분야별 근거와 실제 주장 충돌·한계·리스크를 검토하세요."
        ) + EVIDENCE_RULES + FINAL_REPORT_RULES + extra_instructions,
    )


    if independent_reasoning:
        if not production_output:
            raise ValueError('Independent reasoning requires ProductionReport')
        from independent_reasoning import make_integrator
        integrator = make_integrator(model_client, lambda: list(receipts.values()),
                                     lambda: active_task_text, integrator, peer_review=peer_review)

    # ═══════════════════════════════════════════════════════════════
    # SelectorGroupChat 설정
    # ═══════════════════════════════════════════════════════════════

    # Source text or an expert's prose must not terminate the other experts' work.
    termination = SourceMatchTermination(["Integrator"])

    return SelectorGroupChat(
        participants=[search_expert, economy_expert, market_expert, semi_expert, integrator],
        model_client=model_client,
        termination_condition=termination,
        max_turns=max_turns,
        selector_func=selector,
        custom_message_types=([StructuredMessage[ProductionReport]] if independent_reasoning else
                              [StructuredMessage[final_output_type]] if final_output_type else None),
    )
