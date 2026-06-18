"""
TravelAgent CLI 启动器。

使用重构后的 api.cli 模块启动命令行交互。
用法: python run_cli.py
"""

import sys
import os

# 确保 src 目录在 Python 路径中
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from travel_agent.api.cli import main

if __name__ == "__main__":
    main()
