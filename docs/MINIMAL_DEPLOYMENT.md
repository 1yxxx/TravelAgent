# TravelAgent 最小生产 Demo 部署

目标链路：

`浏览器 -> HTTPS/Caddy -> FastAPI/SSE -> Agent -> MCP 工具 -> DeepSeek/高德 -> 流式响应`

## 1. 云服务器准备

- Linux 云服务器，开放 TCP 80、443 端口
- 安装 Docker Engine 与 Docker Compose Plugin
- 将一个域名 A 记录解析到服务器公网 IP

## 2. 创建运行配置

```bash
cp config.toml.example config.toml
cp .env.example .env
```

编辑 `config.toml`：

- `llm.api_key`：大模型 API Key
- `map.api_key`：高德 Web 服务 Key
- `weather.api_key`：可与 `map.api_key` 共用
- `map.jsapi_key`：高德 Web 端（JS API）Key；未配置时地图降级，聊天主链路仍可运行

编辑 `.env`：

```dotenv
DOMAIN=travel.example.com
```

高德 JS API Key 需要在高德控制台配置与 `DOMAIN` 对应的域名白名单。

## 3. 启动

```bash
docker compose -f compose.yaml -f compose.prod.yaml up -d --build
```

Caddy 会自动申请和续期 HTTPS 证书。首次签发前必须确保域名已解析到当前服务器，且 80、443 端口能从公网访问。

## 4. 验收

```bash
docker compose -f compose.yaml -f compose.prod.yaml ps
docker compose -f compose.yaml -f compose.prod.yaml logs --tail=200
curl -fsS https://travel.example.com/healthz
```

健康检查应返回：

```json
{"status":"ok"}
```

浏览器访问 `https://travel.example.com`，发送“请查询成都今天的天气，必须使用天气工具”，应看到工具执行状态和流式回复。

## 5. 更新版本

```bash
git pull
docker compose -f compose.yaml -f compose.prod.yaml up -d --build
```

## 6. 常用日志

```bash
docker compose -f compose.yaml -f compose.prod.yaml logs -f travel-agent
docker compose -f compose.yaml -f compose.prod.yaml logs -f caddy
```
