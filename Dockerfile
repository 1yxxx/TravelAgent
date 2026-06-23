FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# 先复制依赖清单，充分利用 Docker 构建缓存。
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# 运行时源码。config.toml 不进入镜像，由 Compose 只读挂载。
# 入口脚本
COPY run_server.py run_cli.py ./
COPY src ./src
COPY prompts ./prompts
COPY .storyline ./.storyline
COPY web ./web

RUN mkdir -p /app/travel_data /app/travel_outputs /app/.travel/.server_cache

EXPOSE 8000

# 必须保持单 worker：每个 FastAPI worker 都会在容器内启动 MCP :8002。
# 使用 --factory 模式调用 create_app()，支持重构后的 api.server 模块。
CMD ["uvicorn", "travel_agent.api.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
