"""
FastAPI 服务端 — 多智能体城市规划系统 Web 界面

功能:
  - SSE (Server-Sent Events) 实时流式输出各智能体的运行日志
  - 异步执行工作流，前端实时显示进度
  - 最终报告 Markdown 渲染
"""

import os
import sys
import json
import asyncio
import queue
import threading
import time
import warnings
import logging
import io
from contextlib import redirect_stdout

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from state import AgentState
from agents.case_query_agent import case_query_agent_parallel


# ── 抑制第三方库噪音 ──────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
logging.getLogger("bs4.dammit").setLevel(logging.ERROR)

# ── matplotlib 非交互后端（服务器不弹窗）──────────────────────
import matplotlib
matplotlib.use("Agg")

app = FastAPI(title="Urban Planning Multi-Agent System")

# 静态文件
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "web", "static")), name="static")
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")

# ── 全局任务管理 ─────────────────────────────────────────────
_tasks: dict = {}   # task_id -> { "queue": Queue, "result": dict|None, "status": str }
_cancel_events: dict = {}  # task_id -> threading.Event


# ══════════════════════════════════════════════════════════════
# 路由
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回主页 HTML"""
    html_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str):
    """取消正在运行的任务"""
    if task_id not in _tasks:
        return JSONResponse({"error": "task not found"}, status_code=404)
    if task_id in _cancel_events:
        _cancel_events[task_id].set()
    task = _tasks[task_id]
    task["status"] = "cancelled"
    task["queue"].put({"type": "status", "agent": "system", "text": "⏹️ 用户已终止分析流程"})
    return JSONResponse({"status": "cancelled"})


@app.post("/api/run")
async def start_run(request: Request):
    """启动一次分析任务，返回 task_id"""
    body = await request.json()
    user_query = body.get("query", "").strip()
    if not user_query:
        return JSONResponse({"error": "query 不能为空"}, status_code=400)

    task_id = f"task_{int(time.time()*1000)}"
    msg_queue: queue.Queue = queue.Queue()

    _tasks[task_id] = {
        "queue": msg_queue,
        "result": None,
        "status": "running",
    }

    # 在后台线程运行工作流（因为 LangGraph 是同步阻塞的）
    t = threading.Thread(target=_run_workflow, args=(task_id, user_query, msg_queue), daemon=True)
    t.start()

    return JSONResponse({"task_id": task_id})


@app.get("/api/stream/{task_id}")
async def stream(task_id: str):
    """SSE 端点：实时推送工作流日志"""
    if task_id not in _tasks:
        return JSONResponse({"error": "task not found"}, status_code=404)

    return StreamingResponse(_sse_generator(task_id), media_type="text/event-stream")


@app.get("/api/status/{task_id}")
async def status(task_id: str):
    """查询任务状态"""
    if task_id not in _tasks:
        return JSONResponse({"error": "task not found"}, status_code=404)

    task = _tasks[task_id]
    resp = {"status": task["status"]}
    if task["result"] is not None:
        resp["result"] = task["result"]
    return JSONResponse(resp)


# ══════════════════════════════════════════════════════════════
# SSE 生成器
# ══════════════════════════════════════════════════════════════

async def _sse_generator(task_id: str):
    """把 queue 里的消息包装成 SSE data 帧"""
    task = _tasks[task_id]
    q = task["queue"]

    while True:
        # 非阻塞地从队列取消息
        try:
            msg = q.get_nowait()
        except queue.Empty:
            # 如果任务已结束且队列空了，关闭 SSE
            if task["status"] != "running":
                # 发送最终结果
                if task["result"]:
                    yield _sse_data("result", task["result"])
                yield _sse_data("done", {"status": task["status"]})
                return
            await asyncio.sleep(0.1)
            continue

        yield _sse_data("log", msg)


def _sse_data(event: str, data) -> str:
    """格式化 SSE 帧"""
    payload = json.dumps(data, ensure_ascii=False) if isinstance(data, (dict, list)) else json.dumps({"text": str(data)}, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


# ══════════════════════════════════════════════════════════════
# 工作流执行（在后台线程中运行）
# ══════════════════════════════════════════════════════════════

# 工作流各节点及其展示标签
WORKFLOW_STEPS = [
    ("scenario_deconstruction", "情景解构", "分析本地数据与情境"),
    ("case_query_parallel", "案例查询 (并行)", "并行搜索三个核心问题的全球案例"),
    ("gap_analysis", "差异分析", "对比全球经验与本地情境"),
    ("evaluation", "方案评审", "评估规划方案质量"),
    ("generate_report", "生成报告", "撰写最终规划方案"),
]


def _run_workflow(task_id: str, user_query: str, msg_queue: queue.Queue):
    """在线程中执行工作流，支持真正的取消 + checkpoint"""
    task = _tasks[task_id]

    # 创建取消事件
    cancel_event = threading.Event()
    _cancel_events[task_id] = cancel_event

    def _check_cancelled():
        if cancel_event.is_set():
            raise InterruptedError("用户取消")

    try:
        msg_queue.put({"type": "status", "agent": "system", "text": f"🚀 系统启动，开始分析: {user_query}"})
        msg_queue.put({"type": "steps", "steps": [
            {"id": s[0], "label": s[1], "desc": s[2], "status": "pending"} for s in WORKFLOW_STEPS
        ]})

        initial_state = {
            "user_query": user_query,
            "target_city": "香港",
            "local_context": {},
            "core_problems": [],
            "rewritten_problems": [],
            "case_results": {},
            "gap_analysis": {},
            "adaptation_plan": "",
            "evaluation_scores": {},
            "feedback": "",
            "iteration_count": 0,
            "final_report": "",
            "is_complete": False
        }

        from agents.scenario_agent import scenario_deconstruction_agent
        from agents.case_query_agent import case_query_agent
        from agents.gap_analysis_agent import gap_analysis_agent
        from agents.evaluation_agent import evaluation_agent
        from agents.feedback import feedback_loop
        from agents.report_generator import generate_final_report
        from router import should_continue
        from checkpoint import CheckpointManager

        ckpt = CheckpointManager()
        msg_queue.put({"type": "status", "agent": "system",
                       "text": f"📁 Run ID: {ckpt.run_id}（如中断可恢复）"})

        state = dict(initial_state)

        # ── 按步骤执行，每步前检查取消，每步后保存 checkpoint ──

        # Step 1
        _check_cancelled()
        state = _run_step(task_id, msg_queue, "scenario_deconstruction",
                          lambda: scenario_deconstruction_agent(state), cancel_event)
        ckpt.save(state, "scenario_deconstruction")

        # Step 2: 并行处理三个案例查询
        _check_cancelled()
        step_id = "case_query_parallel"
        def _case_query_parallel():
            s = dict(state)  # 浅拷贝状态
            # 并行处理三个问题
            results = case_query_agent_parallel(s, [0, 1, 2])

            # 构建case_results字典
            case_results = {**s.get("case_results", {})}
            for idx, cases in results.items():
                case_results[f"problem_{idx+1}"] = cases

            # 更新状态
            s["case_results"] = case_results
            return s
        state = _run_step(task_id, msg_queue, step_id, _case_query_parallel, cancel_event)
        ckpt.save(state, step_id)

        # Step 3
        _check_cancelled()
        state = _run_step(task_id, msg_queue, "gap_analysis",
                          lambda: gap_analysis_agent(state), cancel_event)
        ckpt.save(state, "gap_analysis")

        # 评审循环
        max_iterations = 3
        for iteration in range(max_iterations):
            _check_cancelled()
            state = _run_step(task_id, msg_queue, "evaluation",
                              lambda: evaluation_agent(state), cancel_event)
            ckpt.save(state, "evaluation")

            decision = should_continue(state)
            if decision == "generate_report":
                break
            else:
                msg_queue.put({"type": "status", "agent": "evaluation",
                               "text": f"🔄 方案需要改进，启动第 {iteration+1} 轮反馈..."})
                _check_cancelled()
                state = _run_step(task_id, msg_queue, "feedback_loop",
                                  lambda: feedback_loop(state), cancel_event)
                ckpt.save(state, "feedback_loop")

        # Step 5
        _check_cancelled()
        state = _run_step(task_id, msg_queue, "generate_report",
                          lambda: generate_final_report(state), cancel_event)
        ckpt.save(state, "generate_report")

        # 完成
        result = {
            "final_report": state.get("final_report", ""),
            "core_problems": state.get("core_problems", []),
            "matched_area": state.get("local_context", {}).get("matched_area", ""),
            "evaluation_scores": state.get("evaluation_scores", {}),
            "adaptation_plan": state.get("adaptation_plan", ""),
        }

        task["result"] = result
        task["status"] = "done"
        msg_queue.put({"type": "status", "agent": "system", "text": "✅ 分析完成！"})

    except InterruptedError:
        task["status"] = "cancelled"
        msg_queue.put({"type": "status", "agent": "system", "text": "⏹️ 分析已终止"})

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        msg_queue.put({"type": "error", "agent": "system", "text": f"❌ 错误: {str(e)}\n{error_msg}"})
        task["status"] = "error"
        task["result"] = {"error": str(e)}

    finally:
        _cancel_events.pop(task_id, None)


def _run_step(task_id: str, msg_queue: queue.Queue, step_id: str, fn, cancel_event=None):
    """执行一个步骤，支持取消检测"""
    if cancel_event and cancel_event.is_set():
        raise InterruptedError("用户取消")

    msg_queue.put({"type": "step_start", "step_id": step_id})

    capture = _StreamCapture(msg_queue, step_id, cancel_event)
    old_stdout = sys.stdout
    sys.stdout = capture

    try:
        # 在子线程中执行，主线程轮询 cancel_event
        result_container = [None]
        error_container = [None]

        def _target():
            try:
                result_container[0] = fn()
            except Exception as e:
                error_container[0] = e

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()

        # 每 0.5 秒检查一次取消状态
        while worker.is_alive():
            if cancel_event and cancel_event.is_set():
                # 无法强杀线程，但标记状态并退出等待
                msg_queue.put({"type": "step_error", "step_id": step_id, "text": "已终止"})
                sys.stdout = old_stdout
                raise InterruptedError("用户取消")
            worker.join(timeout=0.5)

        if error_container[0] is not None:
            raise error_container[0]

        result = result_container[0]

    except InterruptedError:
        msg_queue.put({"type": "step_error", "step_id": step_id, "text": "已终止"})
        raise
    except Exception as e:
        msg_queue.put({"type": "step_error", "step_id": step_id, "text": str(e)})
        raise
    finally:
        sys.stdout = old_stdout

    msg_queue.put({"type": "step_done", "step_id": step_id})
    return result


class _StreamCapture(io.TextIOBase):
    """自定义 stdout，把 print 输出实时推送到消息队列，支持取消检测"""

    def __init__(self, msg_queue: queue.Queue, step_id: str, cancel_event=None):
        self._queue = msg_queue
        self._step_id = step_id
        self._buffer = ""
        self._cancel = cancel_event

    def write(self, text: str):
        if not text:
            return 0
        # 每次 print 时检查取消
        if self._cancel and self._cancel.is_set():
            raise InterruptedError("用户取消")
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._queue.put({
                    "type": "log",
                    "step_id": self._step_id,
                    "text": line,
                })
        return len(text)

    def flush(self):
        if self._buffer.strip():
            self._queue.put({
                "type": "log",
                "step_id": self._step_id,
                "text": self._buffer.strip(),
            })
            self._buffer = ""


# ══════════════════════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("🌐 启动 Web 界面: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
