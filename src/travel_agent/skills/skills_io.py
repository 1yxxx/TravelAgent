"""将 Markdown Skill 发现结果转换为 LangChain Tool。"""

import aiofiles
from pathlib import Path
from skillkit import SkillManager
from skillkit.integrations.langchain import create_langchain_tools


async def load_skills(
    skill_dir: str = ".storyline/skills"
):
    """
    扫描 Skill 目录并返回 Agent 可直接注册的 Tool 列表。

    Skill 的元数据和工作流来自各目录中的 SKILL.md；此函数只承担
    skillkit 与 LangChain 之间的适配，不负责决定模型何时调用 Skill。
    """
    # 发现并解析目录中的 Markdown Skill。
    manager = SkillManager(skill_dir=skill_dir)
    await manager.adiscover()

    # 转为 LangChain Tool 后，Skill 与普通 MCP Tool 一起暴露给 ReAct Agent。
    tools = create_langchain_tools(manager)
    return tools
