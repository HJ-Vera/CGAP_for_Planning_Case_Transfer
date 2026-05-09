"""
FastAPI 服务端 — 多智能体城市规划系统 Web 界面 (异步版)

功能:
  - SSE (Server-Sent Events) 实时流式输出各智能体的运行日志
  - 异步执行工作流，前端实时显示进度
  - 最终报告 Markdown 渲染
"""

import os
import sys
import asyncio
import json
import queue
import time
import warnings
import logging
import io
import contextvars

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from state import AgentState

import langsmith

os.environ["MPLBACKEND"] = "Agg"
import matplotlib                          # noqa: E402
matplotlib.use("Agg")

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
logging.getLogger("bs4.dammit").setLevel(logging.ERROR)

app = FastAPI(title="Urban Planning Multi-Agent System")

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "web", "static")),
    name="static",
)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")

_tasks: dict = {}
_cancel_events: dict = {}
_active_captures: contextvars.ContextVar = contextvars.ContextVar("active_capture", default=None)


class _StreamCapture:
    """异步 stdout 捕获器，通过 contextvars 路由 print 输出到 SSE 队列"""

    def __init__(self, msg_queue: queue.Queue, step_id: str, cancel_event=None):
        self._queue = msg_queue
        self._step_id = step_id
        self._buffer = ""
        self._cancel = cancel_event

    def receive(self, text: str) -> int:
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


class _RoutedStdout(io.TextIOBase):
    """全局 sys.stdout 替换，通过 contextvars 路由到当前任务捕获器"""

    def __init__(self, fallback: io.TextIOBase):
        self._fallback = fallback

    def write(self, text: str) -> int:
        cap = _active_captures.get(None)
        if cap is not None:
            return cap.receive(text)
        return self._fallback.write(text)

    def flush(self):
        cap = _active_captures.get(None)
        if cap is not None:
            cap.flush()
        else:
            self._fallback.flush()

    @property
    def encoding(self):
        return getattr(self._fallback, "encoding", "utf-8")

    def fileno(self):
        return self._fallback.fileno()


_real_stdout = sys.stdout
sys.stdout = _RoutedStdout(_real_stdout)


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
    task["queue"].put({"type": "status", "agent": "system", "text": "\u23f9\ufe0f \u7528\u6237\u5df2\u7ec8\u6b62\u5206\u6790\u6d41\u7a0b"})
    return JSONResponse({"status": "cancelled"})


@app.post("/api/run")
async def start_run(request: Request):
    body = await request.json()
    user_query = body.get("query", "").strip()
    if not user_query:
        return JSONResponse({"error": "query \u4e0d\u80fd\u4e3a\u7a7a"}, status_code=400)

    task_id = f"task_{int(time.time()*1000)}"
    msg_queue: queue.Queue = queue.Queue()

    _tasks[task_id] = {
        "queue": msg_queue,
        "result": None,
        "status": "running",
    }

    asyncio.create_task(_run_workflow(task_id, user_query, msg_queue))

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


def _send_agent_result(msg_queue: queue.Queue, step_id: str, title: str, markdown: str):
    """向 SSE 队列推送智能体的 markdown 中间结果"""
    if not markdown:
        print(f"⚠️ [server] _send_agent_result: '{title}' (step={step_id}) 内容为空，跳过渲染")
        return
    msg_queue.put({
        "type": "agent_result",
        "step_id": step_id,
        "title": title,
        "markdown": markdown,
    })


WORKFLOW_STEPS = [
    ("scenario_deconstruction", "\u60c5\u666f\u89e3\u6784",     "\u5206\u6790\u672c\u5730\u6570\u636e\u4e0e\u60c5\u5883"),
    ("case_query_1",            "\u6848\u4f8b\u67e5\u8be2 #1",  "\u641c\u7d22\u7b2c\u4e00\u4e2a\u6838\u5fc3\u95ee\u9898\u7684\u5168\u7403\u6848\u4f8b"),
    ("case_query_2",            "\u6848\u4f8b\u67e5\u8be2 #2",  "\u641c\u7d22\u7b2c\u4e8c\u4e2a\u6838\u5fc3\u95ee\u9898\u7684\u5168\u7403\u6848\u4f8b"),
    ("case_query_3",            "\u6848\u4f8b\u67e5\u8be2 #3",  "\u641c\u7d22\u7b2c\u4e09\u4e2a\u6838\u5fc3\u95ee\u9898\u7684\u5168\u7403\u6848\u4f8b"),
    ("gap_analysis",            "\u5dee\u5f02\u5206\u6790",     "\u5bf9\u6bd4\u5168\u7403\u7ecf\u9a8c\u4e0e\u672c\u5730\u60c5\u5883"),
    ("evaluation",              "\u65b9\u6848\u8bc4\u5ba1",     "\u8bc4\u4f30\u89c4\u5212\u65b9\u6848\u8d28\u91cf"),
    ("generate_report",         "\u751f\u6210\u62a5\u544a",     "\u64b0\u5199\u6700\u7ec8\u89c4\u5212\u65b9\u6848"),
]


async def _run_workflow(task_id: str, user_query: str, msg_queue: queue.Queue):
    task = _tasks[task_id]
    cancel_event = asyncio.Event()
    _cancel_events[task_id] = cancel_event

    async def _check_cancelled():
        if cancel_event.is_set():
            raise InterruptedError("\u7528\u6237\u53d6\u6d88")

    with langsmith.trace(
        name="urban_planning_analysis",
        run_type="chain",
        metadata={
            "user_query": user_query,
            "task_id": task_id,
            "entry_point": "web",
        },
        tags=["urban-planning", "web"],
    ):
        await _run_workflow_inner(task_id, user_query, msg_queue, task, cancel_event, _check_cancelled)


async def _run_workflow_inner(task_id, user_query, msg_queue, task, cancel_event, _check_cancelled):
    try:
        msg_queue.put({"type": "status", "agent": "system",
                       "text": f"\U0001f680 \u7cfb\u7edf\u542f\u52a8\uff0c\u5f00\u59cb\u5206\u6790: {user_query}"})
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

        completed_steps: set = set()

        def _fresh_state():
            return {
                "user_query": user_query,
                "target_city": "\u9999\u6e2f",
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
                               "text": f"\U0001f504 \u4ece\u65ad\u70b9\u6062\u590d run_id={ckpt.run_id}\uff0c"
                                       f"\u8df3\u8fc7: {sorted(completed_steps)}"})
            else:
                ckpt = CheckpointManager()
                state = _fresh_state()
        else:
            ckpt = CheckpointManager()
            state = _fresh_state()

        msg_queue.put({"type": "status", "agent": "system",
                       "text": f"\U0001f4c1 Run ID: {ckpt.run_id}"})

        await _check_cancelled()
        if "scenario_deconstruction" not in completed_steps:
            state = await _run_step_async(
                task_id, msg_queue, "scenario_deconstruction",
                lambda: scenario_deconstruction_agent(state), cancel_event,
                on_done=lambda s: _send_agent_result(
                    msg_queue, "scenario_deconstruction", "情景解构分析",
                    s.get("local_context", {}).get("full_response", "")))
            ckpt.save(state, "scenario_deconstruction")
        else:
            msg_queue.put({"type": "step_done", "step_id": "scenario_deconstruction"})
            msg_queue.put({"type": "status", "agent": "scenario_deconstruction",
                           "text": "\u23e9 \u60c5\u666f\u89e3\u6784\u5df2\u5b8c\u6210\uff0c\u8df3\u8fc7"})

        await _check_cancelled()
        pending_indices = [i for i in range(3)
                           if f"case_query_{i + 1}" not in completed_steps]

        if pending_indices:
            msg_queue.put({"type": "status", "agent": "case_query",
                           "text": f"\U0001f680 \u5e76\u884c\u542f\u52a8 {len(pending_indices)} \u4e2a\u6848\u4f8b\u67e5\u8be2\u667a\u80fd\u4f53..."})

            state_snapshot = dict(state)

            async def _make_case_task(idx: int):
                step_id = f"case_query_{idx + 1}"

                async def _fn():
                    cases = await case_query_agent(state_snapshot, idx)
                    return {f"problem_{idx + 1}": cases}

                def _on_case_done(r):
                    cl = r.get(f"problem_{idx + 1}", [])
                    if cl and cl[0].get("global_summary"):
                        _send_agent_result(msg_queue, step_id,
                                           f"案例查询 #{idx + 1} — 全球案例总结",
                                           cl[0]["global_summary"])

                result = await _run_step_async(task_id, msg_queue, step_id, _fn, cancel_event,
                                                on_done=_on_case_done)
                return step_id, result

            tasks_results = await asyncio.gather(
                *[_make_case_task(i) for i in pending_indices]
            )
            for step_id, partial_results in tasks_results:
                state["case_results"] = {
                    **state.get("case_results", {}),
                    **partial_results,
                }
                ckpt.save(state, step_id)
        else:
            for i in range(1, 4):
                msg_queue.put({"type": "step_done", "step_id": f"case_query_{i}"})
            msg_queue.put({"type": "status", "agent": "case_query",
                           "text": "\u23e9 \u6240\u6709\u6848\u4f8b\u67e5\u8be2\u5df2\u5b8c\u6210\uff0c\u8df3\u8fc7"})

        msg_queue.put({"type": "status", "agent": "case_query",
                       "text": f"\u2705 \u6848\u4f8b\u67e5\u8be2\u9636\u6bb5\u5b8c\u6210\uff0c\u5171\u6536\u96c6 "
                               f"{sum(len(v) for v in state.get('case_results', {}).values())} \u4e2a\u6848\u4f8b"})

        await _check_cancelled()
        if "gap_analysis" not in completed_steps:
            state = await _run_step_async(
                task_id, msg_queue, "gap_analysis",
                lambda: gap_analysis_agent(state), cancel_event,
                on_done=lambda s: _send_agent_result(
                    msg_queue, "gap_analysis", "差异分析 — 适应方案",
                    s.get("adaptation_plan", "")))
            ckpt.save(state, "gap_analysis")
        else:
            msg_queue.put({"type": "step_done", "step_id": "gap_analysis"})
            msg_queue.put({"type": "status", "agent": "gap_analysis",
                           "text": "\u23e9 \u5dee\u5f02\u5206\u6790\u5df2\u5b8c\u6210\uff0c\u8df3\u8fc7"})

        max_iterations = 3
        for iteration in range(max_iterations):
            await _check_cancelled()
            state = await _run_step_async(task_id, msg_queue, "evaluation",
                                          lambda: evaluation_agent(state), cancel_event)
            ckpt.save(state, "evaluation")

            decision = should_continue(state)
            if decision == "generate_report":
                break
            msg_queue.put({"type": "status", "agent": "evaluation",
                           "text": f"\U0001f504 \u65b9\u6848\u9700\u8981\u6539\u8fdb\uff0c\u542f\u52a8\u7b2c {iteration + 1} \u8f6e\u53cd\u9988..."})
            await _check_cancelled()
            state = await _run_step_async(task_id, msg_queue, "feedback_loop",
                                          lambda: feedback_loop(state), cancel_event)
            ckpt.save(state, "feedback_loop")

        await _check_cancelled()
        state = await _run_step_async(task_id, msg_queue, "generate_report",
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
        msg_queue.put({"type": "status", "agent": "system", "text": "\u2705 \u5206\u6790\u5b8c\u6210\uff01"})

    except InterruptedError:
        task["status"] = "cancelled"
        msg_queue.put({"type": "status", "agent": "system", "text": "\u23f9\ufe0f \u5206\u6790\u5df2\u7ec8\u6b62"})

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        msg_queue.put({"type": "error", "agent": "system",
                       "text": f"\u274c \u9519\u8bef: {str(e)}\n{error_msg}"})
        task["status"] = "error"
        task["result"] = {"error": str(e)}

    finally:
        _cancel_events.pop(task_id, None)


async def _run_step_async(task_id: str, msg_queue: queue.Queue, step_id: str, fn, cancel_event=None, on_done=None):
    """异步执行单个步骤，通过 contextvars 路由 print 输出到 SSE 队列"""
    if cancel_event and cancel_event.is_set():
        raise InterruptedError("\u7528\u6237\u53d6\u6d88")

    msg_queue.put({"type": "step_start", "step_id": step_id})

    capture = _StreamCapture(msg_queue, step_id, cancel_event)
    token = _active_captures.set(capture)
    try:
        result = await fn()
    finally:
        _active_captures.reset(token)
        capture.flush()

    # 在 step_done 之前推送中间结果，确保前端在步骤切换前看到 markdown
    if on_done:
        on_done(result)

    msg_queue.put({"type": "step_done", "step_id": step_id})
    return result


if __name__ == "__main__":
    import uvicorn
    print("\U0001f310 \u542f\u52a8 Web \u754c\u9762: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
