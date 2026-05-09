"""
Plan-Execute 模式 — 多智能体城市规划案例查询系统

与 workflow 模式共享 tools/ 和 llm.py，但使用不同的执行模式：
  1. Plan Agent 制定任务计划
  2. Sub Agents 执行各子任务
  3. Plan Agent 评估执行结果，决定是否重新规划
  4. 生成最终报告
"""
