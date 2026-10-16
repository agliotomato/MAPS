"""
macro_agent.prompts — LLM 프롬프트 템플릿 모음

노드별 사용 가이드:
    synthesizer 노드 (종합 분석):
        from macro_agent.prompts import build_macro_chain, build_chain_input

    geo_analyst 노드 (지정학 특화):
        from macro_agent.prompts import build_geo_chain, build_geo_chain_input
"""

from macro_agent.prompts.geo_prompt import (
    build_geo_chain,
    build_geo_chain_input,
    build_geo_prompt,
)
from macro_agent.prompts.macro_prompt import (
    build_chain_input,
    build_macro_chain,
    build_macro_prompt,
    format_tool_data,
)

__all__ = [
    # 종합 분석 (macro_prompt)
    "build_macro_prompt",
    "build_macro_chain",
    "build_chain_input",
    "format_tool_data",
    # 지정학 분석 (geo_prompt)
    "build_geo_prompt",
    "build_geo_chain",
    "build_geo_chain_input",
]
