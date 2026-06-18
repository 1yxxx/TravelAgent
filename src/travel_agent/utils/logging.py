"""项目统一日志配置。"""

import logging
from typing import Optional


def setup_logger(name: str = "travel_agent", level: int = logging.INFO) -> logging.Logger:
    """
    创建控制台 Logger。

    已存在 handler 时直接复用，避免模块被多次导入后重复打印同一条日志。
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


# 大多数模块直接导入该单例；需要独立名称时可调用 setup_logger(name)。
logger: logging.Logger = setup_logger()

