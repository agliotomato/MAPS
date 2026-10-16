"""Semiconductor agent package exports.

Keep imports lazy here so utility modules can be reused without creating
experience<->agent circular imports during batch jobs.
"""


def get_tool_inventory():
    from app.agent.semiconductor_tools import get_tool_inventory as _get_tool_inventory

    return _get_tool_inventory()


def run_bounded_semiconductor_react(*args, **kwargs):
    from app.agent.semiconductor_react import run_bounded_semiconductor_react as _run

    return _run(*args, **kwargs)


def render_bounded_react_summary(*args, **kwargs):
    from app.agent.semiconductor_react import render_bounded_react_summary as _render

    return _render(*args, **kwargs)


__all__ = [
    "get_tool_inventory",
    "run_bounded_semiconductor_react",
    "render_bounded_react_summary",
]
