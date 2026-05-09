"""
FastAPI 服务端 — 多智能体城市规划系统 Web 界面

功能:
  - SSE (Server-Sent Events) 实时流式输出各智能体的运行日志
  - 异步执行工作流，前端实时显示进度
  - 最终报告 Markdown 渲染

并行改动说明:
  - _ThreadRoutedStdout：全局 sys.stdout 替换，通过 threading.local 将
    print() 输出路由到当前线程自己的 _StreamCapture，解决并行时 stdout 竞争。
  - _run_workflow 中三个 case_query 步骤改为 ThreadPoolExecutor 并行执行。
"""

import os
import sys

# ══════════════════════════════════════════════════════════════
# CRITICAL: Force non-GUI matplotlib backend BEFORE any other
# import that might trigger matplotlib's backend auto-detection.
# On Windows the default is TkAgg which uses tkinter — tkinter
# is not thread-safe and will crash when worker threads try to
# garbage-collect its objects ("main thread is not in main loop").
# Setting the env var here ensures every subsequent import of
# matplotlib (including transitive ones inside agents/tools)
# sees the override before the backend is locked in.
# ══════════════════════════════════════════════════════════════
os.environ["MPLBACKEND"] = "Agg"          # env var: catches transitive imports
import matplotlib                          # noqa: E402
matplotlib.use("Agg")                     # explicit call: belt-and-suspenders

import json
import asyncio
import queue
import threading
import time
import warnings
import logging
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from state import AgentState

# ── 抑制第三方库噪音 ──────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
logging.getLogger("bs4.dammit").setLevel(logging.ERROR)

app = FastAPI(title="Urban Planning Multi-Agent System")

# 静态文件
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "web", "static")),
    name="static",
)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")

# ── 全局任务管理 ─────────────────────────────────────────────
_tasks: dict = {}        # task_id -> { "queue": Queue, "result": dict|None, "status": str }
_cancel_events: dict = {}  # task_id -> threading.Event


# ══════════════════════════════════════════════════════════════
# 线程安全的 stdout 路由
# ══════════════════════════════════════════════════════════════

_tls = threading.local()   # 每个线程独立的 active_capture 槽位
_real_stdout = sys.stdout  # 保存真实 stdout（模块加载时）


class _ThreadRoutedStdout(io.TextIOBase):
    """
    替换 sys.stdout 的全局对象。
    每次 write() 时查看当前线程的 _tls.active_capture：
      - 若已注册 → 交给该线程的 _StreamCapture 处理（发送到 SSE 队列）
      - 否则 → 写入原始 stdout（正常终端输出）

    这样三个并行 case_query 线程各自持有独立的 capture，互不干扰。
    """

    def __init__(self, fallback: io.TextIOBase):
        self._fallback = fallback

    def write(self, text: str) -> int:
        cap = getattr(_tls, "active_capture", None)
        if cap is not None:
            return cap.receive(text)
        return self._fallback.write(text)

    def flush(self):
        cap = getattr(_tls, "active_capture", None)
        if cap is not None:
            cap.flush()
        else:
            self._fallback.flush()

    # 让 logging 等库能正常判断
    @property
    def encoding(self):
        return getattr(self._fallback, "encoding", "utf-8")

    def fileno(self):
        return self._fallback.fileno()


# 安装全局路由（模块加载时执行一次）
sys.stdout = _ThreadRoutedStdout(_real_stdout)


class _StreamCapture:
    """
    线程级 stdout 捕获器。
    通过 _tls.active_capture 注册到当前线程后，该线程的所有 print() 输出
    都会经由 _ThreadRoutedStdout.write() → self.receive() 进入 SSE 队列。
    """

    def __init__(self, msg_queue: queue.Queue, step_id: str, cancel_event=None):
        self._queue = msg_queue
        self._step_id = step_id
        self._buffer = ""
        self._cancel = cancel_event

    def receive(self, text: str) -> int:
        """由 _ThreadRoutedStdout 调用，处理原始 print 输出"""
        if not text:
            return 0
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
# 路由
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str):
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

    t = threading.Thread(
        target=_run_workflow, args=(task_id, user_query, msg_queue), daemon=True
    )
    t.start()

    return JSONResponse({"task_id": task_id})


@app.get("/api/stream/{task_id}")
async def stream(task_id: str):
    if task_id not in _tasks:
        return JSONResponse({"error": "task not found"}, status_code=404)
    return StreamingResponse(_sse_generator(task_id), media_type="text/event-stream")


@app.get("/api/status/{task_id}")
async def status(task_id: str):
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
    task = _tasks[task_id]
    q = task["queue"]

    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            if task["status"] != "running":
                if task["result"]:
                    yield _sse_data("result", task["result"])
                yield _sse_data("done", {"status": task["status"]})
                return
            await asyncio.sleep(0.1)
            continue
        yield _sse_data("log", msg)


def _sse_data(event: str, data) -> str:
    payload = (
        json.dumps(data, ensure_ascii=False)
        if isinstance(data, (dict, list))
        else json.dumps({"text": str(data)}, ensure_ascii=False)
    )
    return f"event: {event}\ndata: {payload}\n\n"


# ══════════════════════════════════════════════════════════════
# 工作流步骤定义
# ══════════════════════════════════════════════════════════════

WORKFLOW_STEPS = [
    ("scenario_deconstruction", "情景解构",     "分析本地数据与情境"),
    ("case_query_1",            "案例查询 #1",  "搜索第一个核心问题的全球案例"),
    ("case_query_2",            "案例查询 #2",  "搜索第二个核心问题的全球案例"),
    ("case_query_3",            "案例查询 #3",  "搜索第三个核心问题的全球案例"),
    ("gap_analysis",            "差异分析",     "对比全球经验与本地情境"),
    ("evaluation",              "方案评审",     "评估规划方案质量"),
    ("generate_report",         "生成报告",     "撰写最终规划方案"),
]


# ══════════════════════════════════════════════════════════════
# 工作流执行（后台线程）
# ══════════════════════════════════════════════════════════════

def _run_workflow(task_id: str, user_query: str, msg_queue: queue.Queue):
    """在线程中执行工作流，支持取消 + checkpoint + 并行 case_query"""
    task = _tasks[task_id]
    cancel_event = threading.Event()
    _cancel_events[task_id] = cancel_event

    def _check_cancelled():
        if cancel_event.is_set():
            raise InterruptedError("用户取消")

    try:
        msg_queue.put({"type": "status", "agent": "system",
                       "text": f"🚀 系统启动，开始分析: {user_query}"})
        msg_queue.put({"type": "steps", "steps": [
            {"id": s[0], "label": s[1], "desc": s[2], "status": "pending"}
            for s in WORKFLOW_STEPS
        ]})

        from agents.scenario_agent import scenario_deconstruction_agent
        from agents.case_query_agent import case_query_agent
        from agents.gap_analysis_agent import gap_analysis_agent
        from agents.evaluation_agent import evaluation_agent
        from agents.feedback import feedback_loop
        from agents.report_generator import generate_final_report
        from router import should_continue
        from checkpoint import CheckpointManager

        # ── Checkpoint 恢复检测 ─────────────────────────────────────
        completed_steps: set = set()

        def _fresh_state():
            return {
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
                "is_complete": False,
            }

        existing = CheckpointManager.find_latest_run(user_query)
        if existing:
            _, saved_state = existing.load_latest()
            done = existing.get_completed_steps()
            if saved_state and "generate_report" not in done:
                state = saved_state
                completed_steps = set(done)
                ckpt = existing
                msg_queue.put({"type": "status", "agent": "system",
                               "text": f"🔄 从断点恢复 run_id={ckpt.run_id}，"
                                       f"跳过: {sorted(completed_steps)}"})
            else:
                ckpt = CheckpointManager()
                state = _fresh_state()
        else:
            ckpt = CheckpointManager()
            state = _fresh_state()

        msg_queue.put({"type": "status", "agent": "system",
                       "text": f"📁 Run ID: {ckpt.run_id}"})

        # ── Step 1: 情景解构 ──────────────────────────────────────
        _check_cancelled()
        if "scenario_deconstruction" not in completed_steps:
            state = _run_step(task_id, msg_queue, "scenario_deconstruction",
                              lambda: scenario_deconstruction_agent(state), cancel_event)
            ckpt.save(state, "scenario_deconstruction")
        else:
            msg_queue.put({"type": "step_done", "step_id": "scenario_deconstruction"})
            msg_queue.put({"type": "status", "agent": "scenario_deconstruction",
                           "text": "⏩ 情景解构已完成，跳过"})

        # ── Step 2: 三个案例查询并行执行 ─────────────────────────
        _check_cancelled()
        pending_indices = [i for i in range(3)
                           if f"case_query_{i + 1}" not in completed_steps]

        if pending_indices:
            msg_queue.put({"type": "status", "agent": "case_query",
                           "text": f"🚀 并行启动 {len(pending_indices)} 个案例查询智能体..."})

            # 每个任务拍一份 state 快照（只读），避免并发修改同一对象
            state_snapshot = dict(state)

            def _make_case_task(idx: int):
                """返回一个可供 executor.submit 调用的函数"""
                step_id = f"case_query_{idx + 1}"

                def _fn():
                    """在 worker 线程中运行，返回局部 case_results"""
                    cases = case_query_agent(state_snapshot, idx)
                    return {f"problem_{idx + 1}": cases}

                def _task():
                    result = _run_step(task_id, msg_queue, step_id, _fn, cancel_event)
                    return step_id, result   # result = {"problem_N": [...]}

                return _task

            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="case_query") as executor:
                futures = {
                    executor.submit(_make_case_task(i)): i
                    for i in pending_indices
                }
                for future in as_completed(futures):
                    _check_cancelled()
                    step_id, partial_results = future.result()  # 若有异常会在此抛出
                    # 合并局部结果到 state
                    state["case_results"] = {
                        **state.get("case_results", {}),
                        **partial_results,
                    }
                    ckpt.save(state, step_id)
        else:
            for i in range(1, 4):
                msg_queue.put({"type": "step_done", "step_id": f"case_query_{i}"})
            msg_queue.put({"type": "status", "agent": "case_query",
                           "text": "⏩ 所有案例查询已完成，跳过"})

        msg_queue.put({"type": "status", "agent": "case_query",
                       "text": f"✅ 案例查询阶段完成，共收集 "
                               f"{sum(len(v) for v in state.get('case_results', {}).values())} 个案例"})

        # ── Step 3: 差异分析 ─────────────────────────────────────
        _check_cancelled()
        if "gap_analysis" not in completed_steps:
            state = _run_step(task_id, msg_queue, "gap_analysis",
                              lambda: gap_analysis_agent(state), cancel_event)
            ckpt.save(state, "gap_analysis")
        else:
            msg_queue.put({"type": "step_done", "step_id": "gap_analysis"})
            msg_queue.put({"type": "status", "agent": "gap_analysis",
                           "text": "⏩ 差异分析已完成，跳过"})

        # ── Step 4: 评审循环 ─────────────────────────────────────
        max_iterations = 3
        for iteration in range(max_iterations):
            _check_cancelled()
            state = _run_step(task_id, msg_queue, "evaluation",
                              lambda: evaluation_agent(state), cancel_event)
            ckpt.save(state, "evaluation")

            decision = should_continue(state)
            if decision == "generate_report":
                break
            msg_queue.put({"type": "status", "agent": "evaluation",
                           "text": f"🔄 方案需要改进，启动第 {iteration + 1} 轮反馈..."})
            _check_cancelled()
            state = _run_step(task_id, msg_queue, "feedback_loop",
                              lambda: feedback_loop(state), cancel_event)
            ckpt.save(state, "feedback_loop")

        # ── Step 5: 生成报告 ─────────────────────────────────────
        _check_cancelled()
        state = _run_step(task_id, msg_queue, "generate_report",
                          lambda: generate_final_report(state), cancel_event)
        ckpt.save(state, "generate_report")

        task["result"] = {
            "final_report":      state.get("final_report", ""),
            "core_problems":     state.get("core_problems", []),
            "matched_area":      state.get("local_context", {}).get("matched_area", ""),
            "evaluation_scores": state.get("evaluation_scores", {}),
            "adaptation_plan":   state.get("adaptation_plan", ""),
        }
        task["status"] = "done"
        msg_queue.put({"type": "status", "agent": "system", "text": "✅ 分析完成！"})

    except InterruptedError:
        task["status"] = "cancelled"
        msg_queue.put({"type": "status", "agent": "system", "text": "⏹️ 分析已终止"})

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        msg_queue.put({"type": "error", "agent": "system",
                       "text": f"❌ 错误: {str(e)}\n{error_msg}"})
        task["status"] = "error"
        task["result"] = {"error": str(e)}

    finally:
        _cancel_events.pop(task_id, None)


# ══════════════════════════════════════════════════════════════
# 步骤执行器（线程安全版）
# ══════════════════════════════════════════════════════════════

def _run_step(
    task_id: str,
    msg_queue: queue.Queue,
    step_id: str,
    fn,
    cancel_event=None,
):
    """
    执行单个步骤。

    线程安全改动：
      不再直接替换 sys.stdout（全局操作，并行时会互相覆盖）。
      改为在 worker 线程内部通过 _tls.active_capture 注册捕获器，
      由模块级 _ThreadRoutedStdout 负责路由 print() 输出。
    """
    if cancel_event and cancel_event.is_set():
        raise InterruptedError("用户取消")

    msg_queue.put({"type": "step_start", "step_id": step_id})

    capture = _StreamCapture(msg_queue, step_id, cancel_event)
    result_container = [None]
    error_container = [None]

    def _target():
        # 在 worker 线程中注册捕获器，_ThreadRoutedStdout 会将
        # 该线程的所有 print() 路由到此 capture
        _tls.active_capture = capture
        try:
            result_container[0] = fn()
        except Exception as e:
            error_container[0] = e
        finally:
            _tls.active_capture = None   # 取消注册，恢复正常输出
            capture.flush()              # 冲刷剩余缓冲

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()

    # 主调线程轮询取消事件，每 0.5s 一次
    while worker.is_alive():
        if cancel_event and cancel_event.is_set():
            msg_queue.put({"type": "step_error", "step_id": step_id, "text": "已终止"})
            raise InterruptedError("用户取消")
        worker.join(timeout=0.5)

    if error_container[0] is not None:
        msg_queue.put({"type": "step_error", "step_id": step_id,
                       "text": str(error_container[0])})
        raise error_container[0]

    msg_queue.put({"type": "step_done", "step_id": step_id})
    return result_container[0]


# ══════════════════════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("🌐 启动 Web 界面: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)