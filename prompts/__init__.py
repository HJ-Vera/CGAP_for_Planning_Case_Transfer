"""
Prompt 模板加载与注入工具

使用 string.Template 语法，prompt 文件中用 $variable 占位，
JSON 中的 { } 大括号无需转义。
"""

import os
from pathlib import Path
from string import Template
from typing import Any, Dict


_PROMPT_ROOT = Path(__file__).parent


def load_prompt(agent: str, name: str, **kwargs: Any) -> str:
    """
    加载 prompt 模板并填充变量。

    参数:
        agent: agent 子目录名, 如 "agents/scenario_agent", "plan_execute/sub_agents"
        name:  prompt 文件名 (不含 .txt 后缀), 如 "01_table_understanding_prompt"
        **kwargs: 模板变量

    返回:
        填充后的 prompt 字符串
    """
    filepath = _PROMPT_ROOT / agent / f"{name}.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    template_text = filepath.read_text(encoding="utf-8").lstrip("\ufeff")

    # 处理 chr(10) 等特殊替换 - 如果原始代码用了 chr(10).join()
    # 这些在传入 kwargs 时就已经是拼接好的字符串，所以直接用 Template

    tmpl = Template(template_text)
    try:
        return tmpl.substitute(**kwargs)
    except KeyError as e:
        missing = str(e).strip("'")
        print(f"  ⚠️ Prompt 模板缺少变量: {missing}, 使用空字符串替代")
        # safe_substitute 不会报错，缺失变量保留原样
        return tmpl.safe_substitute(**kwargs)
