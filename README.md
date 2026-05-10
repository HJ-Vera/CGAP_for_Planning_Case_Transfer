## 项目简介

### **CGAP** 是一个面向城市规划案例迁移的结构化推理系统。

该系统尝试解决：

> “国际成功经验如何在不同城市背景下进行可迁移性分析与本地化适配”

这一问题。

与传统 RAG 或普通 Agent Workflow 不同，CGAP 将规划案例学习拆解为：

- Context Modeling（情景建模） 
- Case Retrieval（案例检索）   
- Gap Analysis（差异对比）
    
三个显式推理阶段。

系统通过中间状态（Intermediate Representations）组织信息流，而非直接依赖单次上下文推理，从而提高复杂规划任务中的证据整合能力与上下文稳定性。


## CGAP Workflow（推理流程）

---
<div align="center"> 
<img width="915" height="492" alt="image" src="https://github.com/user-attachments/assets/05b0b62b-7ecb-4b87-91d6-b238dbda3c13" />
</div>   

---

> CGAP 将复杂规划推理拆解为多个结构化阶段, 通过显式信息组织减少上下文噪声与隐式推理负担。
