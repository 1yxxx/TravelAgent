"""
文件上传与 PDF 导出路由。
"""

from __future__ import annotations

import io
from fastapi import UploadFile, File, Form, Body, HTTPException
from fastapi.responses import StreamingResponse

from travel_agent.config import load_settings
from travel_agent.api.html_render import build_itinerary_html
from travel_agent.utils.logging import logger


def register_file_routes(app, config_path: str):
    """向 FastAPI app 注册文件相关路由。"""

    @app.post("/api/upload")
    async def upload_file(file: UploadFile = File(...), city: str = Form("")):
        """示例上传接口：把用户偏好 / 历史行程等 JSON 文件传到 data_dir 下。"""
        cfg = load_settings(config_path)
        data_dir = cfg.project.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        dest = data_dir / file.filename
        content = await file.read()
        dest.write_bytes(content)
        return {"filename": file.filename, "city": city}

    @app.post("/api/export-pdf")
    async def export_pdf(payload: dict = Body(...)):
        """接收行程 JSON，返回可直接用浏览器打印为 PDF 的 HTML 页面。

        请求体格式：
        {
          "itinerary": { "city": "北京", "title": "...", "days": [...] },
          "hotel":     [...],
          "restaurant": [...]
        }
        """
        try:
            html_str = build_itinerary_html(payload)
            city = (payload.get("itinerary") or {}).get("city", "旅行")
            filename = f"{city}旅行攻略.pdf"
            return StreamingResponse(
                io.BytesIO(html_str.encode("utf-8")),
                media_type="text/html; charset=utf-8",
                headers={"Content-Disposition": f'inline; filename*=UTF-8\'\'{filename}'},
            )
        except Exception as exc:
            logger.exception("[export-pdf] 失败: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))
