"""
Tool: request_travel_info — LLM calls this when the user's travel request is
missing key information (destination, days, budget, etc.). The tool returns a
structured payload that the SSE handler converts into an A2UI form card.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool


@tool("request_travel_info", return_direct=False)
def request_travel_info_tool(
    missing_fields: str = "",
    custom_message: Optional[str] = None,
) -> str:
    """Call this tool when the user's travel request is missing critical
    information that must be collected before planning can begin.

    Args:
        missing_fields: Comma-separated list of missing field keys.
            Valid keys: destination, days, budget, preference.
            Example: "destination,days"
        custom_message: A friendly message shown above the form fields.
            If omitted, a default message is used.

    Returns:
        A JSON string that the frontend will render as an interactive form.
    """
    fields = [f.strip() for f in missing_fields.split(",") if f.strip()]

    # Map field keys to form field configs
    FIELD_TEMPLATES = {
        "destination": {
            "key": "destination", "label": "目的地城市", "field_type": "text",
            "placeholder": "例如：成都、杭州、上海", "required": True,
        },
        "days": {
            "key": "days", "label": "出行天数", "field_type": "number",
            "placeholder": "例如：3", "required": True, "min": 1, "max": 30,
        },
        "budget": {
            "key": "budget", "label": "预算范围", "field_type": "select",
            "options": [
                {"label": "经济实惠（2000以内）", "value": "budget"},
                {"label": "舒适享受（2000-5000）", "value": "mid"},
                {"label": "豪华体验（5000以上）", "value": "luxury"},
            ],
        },
        "preference": {
            "key": "preference", "label": "旅行偏好", "field_type": "textarea",
            "placeholder": "例如：亲子游、美食之旅、历史文化…",
        },
    }

    form_fields = []
    for key in fields:
        if key in FIELD_TEMPLATES:
            form_fields.append(FIELD_TEMPLATES[key])

    # Always include at least destination + days
    if not form_fields:
        form_fields = [
            FIELD_TEMPLATES["destination"],
            FIELD_TEMPLATES["days"],
            FIELD_TEMPLATES["preference"],
        ]

    message = custom_message or "请补充以下信息，以便我为你精准规划行程："

    return json.dumps({
        "__a2ui_form": True,
        "id": "llm_request_info",
        "title": "完善旅行信息",
        "message": message,
        "fields": form_fields,
    }, ensure_ascii=False)
