"""
共享的场景标签识别模块。

本模块提供的 ``_infer_scenario_tags`` 和 ``_extract_text_from_messages``
同时被 ``LayeredTravelAgent`` (orchestration) 和 ``NodeManager`` (agent) 使用，
消除了原先两处各自维护一份相似逻辑的 DRY 违反。
"""

from __future__ import annotations

from typing import Any, List, Set


def _extract_text_from_messages(messages: List[Any]) -> str:
    """统一拍平 LangChain 的字符串/内容块消息，供关键词规则扫描。"""
    chunks: List[str] = []
    for m in messages:
        content = getattr(m, "content", "") or ""
        if isinstance(content, list):
            text = "\n".join(
                (x.get("text") or "") if isinstance(x, dict) else str(x)
                for x in content
            )
            chunks.append(text)
        else:
            chunks.append(str(content))
    return "\n".join(chunks)


def infer_scenario_tags(text: str) -> Set[str]:
    """从用户文本推断人群、预算、偏好和行程长度等场景标签。

    与原先 ``node_manager.py`` 中的版本完全一致，增加了耦合/户外/人文等标签。
    """
    t = (text or "").lower()
    tags: Set[str] = set()

    # 人群类型
    if any(k in t for k in ("亲子", "小孩", "儿童", "带娃", "家庭")):
        tags.add("family")
    if any(k in t for k in ("老人", "老年", "无障碍", "轮椅")):
        tags.add("senior")
    if any(k in t for k in ("情侣", "约会", "蜜月", "浪漫")):
        tags.add("couple")
    if any(k in t for k in ("独行", "一个人", "solo", "单人")):
        tags.add("solo")

    # 预算等级
    if any(k in t for k in ("穷游", "省钱", "预算低", "性价比", "便宜")):
        tags.add("budget")
    if any(k in t for k in ("轻奢", "高端", "豪华", "luxury", "定制")):
        tags.add("luxury")

    # 活动偏好
    if any(k in t for k in ("户外", "徒步", "露营", "登山", "骑行")):
        tags.add("outdoor")
    if any(k in t for k in ("人文", "博物馆", "古镇", "历史", "文化")):
        tags.add("culture")

    # 行程长度
    if any(k in t for k in ("短途", "周边", "周末", "1天", "2天")):
        tags.add("short_trip")
    if any(k in t for k in ("长线", "环线", "跨省", "多城", "5天", "7天", "10天")):
        tags.add("long_trip")

    if any(k in t for k in ("定制", "个性化", "小众", "深度游")):
        tags.add("custom")

    return tags
