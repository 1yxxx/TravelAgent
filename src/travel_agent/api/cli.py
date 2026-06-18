"""
CLI 交互式旅行助手入口。

通过 stdin/stdout 与用户对话，调用 Agent 处理请求。
每个会话复用相同的 build_agent() 工厂和 ClientContext。
"""

from __future__ import annotations

import asyncio
import uuid
import time

from langchain_core.messages import HumanMessage

from travel_agent.agent.factory import build_agent
from travel_agent.config import load_settings, default_config_path
from travel_agent.api.message_utils import normalize_content
from travel_agent.utils.logging import logger


async def run_cli():
    """CLI 交互主循环。"""
    cfg = load_settings(default_config_path())

    session_id = f"cli_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    agent, context = await build_agent(cfg=cfg, session_id=session_id, lang="zh")

    messages = []
    print("智能旅行助手已就绪，输入 /exit 退出。")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            print("再见！")
            break

        messages.append(HumanMessage(content=user_input))

        try:
            invoke_messages, _ = await context.prepare_messages_for_invoke(messages)
            result = await agent.ainvoke({"messages": invoke_messages})
            raw_messages = result["messages"]

            # 提取 AI 回复
            reply = ""
            for m in reversed(raw_messages):
                content = getattr(m, "content", None)
                if content and not str(m.__class__.__name__).startswith("Tool"):
                    reply = normalize_content(content)
                    if reply:
                        break

            print(f"\n助手: {reply}")
        except Exception as exc:
            logger.exception("Agent 调用失败: %s", exc)
            print(f"\n❌ 发生错误：{exc}")


def main():
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
