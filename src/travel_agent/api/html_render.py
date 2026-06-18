"""
行程 HTML 渲染器。

将行程 JSON 渲染为带样式的 HTML 字符串，用于 PDF 导出。
"""

from __future__ import annotations

import datetime
from typing import Dict


DAY_COLORS = ["#4f46e5", "#0891b2", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0284c7"]

CSS = """
@page { size: A4; margin: 20mm 18mm; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "PingFang SC", "Microsoft YaHei", "Hiragino Sans GB", sans-serif;
       font-size: 11pt; color: #1e293b; line-height: 1.6; }
h1 { font-size: 22pt; font-weight: 700; color: #1e293b;
     border-bottom: 3px solid #4f46e5; padding-bottom: 8px; margin-bottom: 6px; }
.subtitle { font-size: 10pt; color: #64748b; margin-bottom: 20px; }
.section-title { font-size: 13pt; font-weight: 600; color: #334155;
                 margin: 22px 0 10px; padding-bottom: 4px;
                 border-bottom: 1px solid #e2e8f0; }
.day-block { margin-bottom: 18px; page-break-inside: avoid; }
.day-label { font-size: 12pt; font-weight: 700; padding: 4px 10px;
             border-radius: 4px; color: #fff; display: inline-block;
             margin-bottom: 8px; }
.spot-row { display: flex; align-items: flex-start; gap: 8px;
            margin-bottom: 7px; padding: 7px 10px;
            background: #f8fafc; border-radius: 6px; }
.spot-num { width: 22px; height: 22px; border-radius: 50%; color: #fff;
            font-size: 9pt; font-weight: 700; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center; }
.spot-name { font-weight: 600; font-size: 10.5pt; }
.spot-meta { font-size: 9pt; color: #64748b; margin-top: 1px; }
.spot-note { font-size: 9pt; color: #7c3aed; margin-top: 2px; font-style: italic; }
.sub-label { font-size: 10pt; font-weight: 600; color: #475569;
             margin: 8px 0 4px; }
.poi-row { display: flex; align-items: flex-start; gap: 8px;
           margin-bottom: 5px; padding: 6px 10px;
           background: #f0fdf4; border-radius: 6px; }
.poi-row.hotel { background: #eff6ff; }
.poi-row.rest  { background: #f0fdf4; }
.poi-name { font-weight: 600; font-size: 10.5pt; }
.poi-meta { font-size: 9pt; color: #64748b; }
table.summary { width: 100%; border-collapse: collapse; margin-top: 8px;
                font-size: 10pt; }
table.summary th { background: #f1f5f9; padding: 6px 10px;
                   text-align: left; font-weight: 600; border: 1px solid #e2e8f0; }
table.summary td { padding: 6px 10px; border: 1px solid #e2e8f0; vertical-align: top; }
.footer { margin-top: 28px; font-size: 9pt; color: #94a3b8;
          border-top: 1px solid #e2e8f0; padding-top: 8px; text-align: right; }
"""


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_itinerary_html(data: dict) -> str:
    """把行程 JSON 渲染成带样式的 HTML 字符串（用于 PDF 转换）。"""
    itinerary = data.get("itinerary") or {}
    hotels = data.get("hotel", [])
    restaurants = data.get("restaurant", [])
    city = itinerary.get("city", "")
    title = itinerary.get("title", "") or (f"{city}旅行攻略" if city else "旅行攻略")
    days = itinerary.get("days", [])

    today = datetime.date.today().strftime("%Y年%m月%d日")

    parts = [
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">',
        f'<style>{CSS}</style></head><body>',
        f'<h1>{esc(title)}</h1>',
        f'<div class="subtitle">生成日期：{today}</div>',
    ]

    # ── 每日行程 ──
    if days:
        parts.append('<div class="section-title">📅 每日行程</div>')
        for di, day in enumerate(days):
            color = DAY_COLORS[di % len(DAY_COLORS)]
            parts.append('<div class="day-block">')
            parts.append(
                f'<div class="day-label" style="background:{color}">'
                f'{esc(day.get("label", "第" + str(di+1) + "天"))}</div>'
            )
            for si, spot in enumerate(day.get("spots") or []):
                name = esc(spot.get("name", ""))
                addr = esc(spot.get("address", ""))
                note = esc(spot.get("note", ""))
                meta_parts = []
                if spot.get("rating"):
                    meta_parts.append(f'⭐{spot["rating"]}')
                if addr:
                    meta_parts.append(f'📍{addr}')
                parts.append(
                    f'<div class="spot-row">'
                    f'<div class="spot-num" style="background:{color}">{si+1}</div>'
                    f'<div><div class="spot-name">{name}</div>'
                    f'{"<div class=spot-meta>" + " · ".join(meta_parts) + "</div>" if meta_parts else ""}'
                    f'{"<div class=spot-note>💡 " + note + "</div>" if note else ""}'
                    f'</div></div>'
                )
            if day.get("hotel"):
                h = day["hotel"]
                parts.append('<div class="sub-label">🏨 住宿</div>')
                parts.append(
                    f'<div class="poi-row hotel">'
                    f'<div><div class="poi-name">{esc(h.get("name",""))}</div>'
                    f'<div class="poi-meta">{"⭐"+str(h["rating"]) if h.get("rating") else ""}'
                    f'{"  📍"+esc(h.get("address","")) if h.get("address") else ""}</div></div></div>'
                )
            if day.get("meals"):
                parts.append('<div class="sub-label">🍜 餐厅</div>')
                for meal in day["meals"]:
                    parts.append(
                        f'<div class="poi-row rest">'
                        f'<div><div class="poi-name">{esc(meal.get("name",""))}</div>'
                        f'<div class="poi-meta">{"🍽️"+esc(meal.get("cuisine","")) if meal.get("cuisine") else ""}'
                        f'{"  ⭐"+str(meal["rating"]) if meal.get("rating") else ""}'
                        f'{"  💰¥"+str(meal.get("cost","")) if meal.get("cost") else ""}'
                        f'{"  📍"+esc(meal.get("address","")) if meal.get("address") else ""}</div></div></div>'
                    )
            parts.append('</div>')

    # ── 住宿汇总 ──
    if hotels:
        parts.append('<div class="section-title">🏨 住宿推荐</div>')
        parts.append('<table class="summary"><tr><th>酒店名称</th><th>评分</th><th>地址</th></tr>')
        for h in hotels:
            parts.append(
                f'<tr><td>{esc(h.get("name",""))}</td>'
                f'<td>{"⭐"+str(h["rating"]) if h.get("rating") else "-"}</td>'
                f'<td>{esc(h.get("address",""))}</td></tr>'
            )
        parts.append('</table>')

    # ── 餐厅汇总 ──
    if restaurants:
        parts.append('<div class="section-title">🍜 餐厅推荐</div>')
        parts.append(
            '<table class="summary"><tr><th>餐厅名称</th><th>菜系</th>'
            '<th>人均</th><th>评分</th><th>地址</th></tr>'
        )
        for r in restaurants:
            parts.append(
                f'<tr><td>{esc(r.get("name",""))}</td>'
                f'<td>{esc(r.get("cuisine",""))}</td>'
                f'<td>{"¥"+str(r.get("cost","")) if r.get("cost") else "-"}</td>'
                f'<td>{"⭐"+str(r["rating"]) if r.get("rating") else "-"}</td>'
                f'<td>{esc(r.get("address",""))}</td></tr>'
            )
        parts.append('</table>')

    parts.append(f'<div class="footer">由智能旅行助手生成 · {today}</div>')
    parts.append('<script>window.onload=function(){window.print();}</script>')
    parts.append('</body></html>')
    return "".join(parts)
