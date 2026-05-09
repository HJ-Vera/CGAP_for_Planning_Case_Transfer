/* ═══════════════════════════════════════════════════════════
   Urban Planning Multi-Agent System — Frontend Logic
   ═══════════════════════════════════════════════════════════ */

let currentTaskId = null;
let isRunning = false;

// Agent display mapping
const AGENT_META = {
  system:                    { icon: "⚙️",  name: "系统" },
  scenario_deconstruction:   { icon: "🔍", name: "情景解构" },
  case_query_1:              { icon: "🌍", name: "案例查询 #1" },
  case_query_2:              { icon: "🌏", name: "案例查询 #2" },
  case_query_3:              { icon: "🌎", name: "案例查询 #3" },
  gap_analysis:              { icon: "🔬", name: "差异分析" },
  evaluation:                { icon: "⚖️",  name: "方案评审" },
  generate_report:           { icon: "📄", name: "生成报告" },
};

// ── Utilities ────────────────────────────────────────────────

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function scrollToBottom() {
  const c = $("#chatContainer");
  requestAnimationFrame(() => c.scrollTop = c.scrollHeight);
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 150) + "px";
}

function formatTime() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}

// ── Sidebar ──────────────────────────────────────────────────

function toggleSidebar() {
  const sb = $("#sidebar");
  sb.classList.toggle("open");
}

// ── Welcome & Examples ───────────────────────────────────────

function fillExample(btn) {
  const input = $("#queryInput");
  input.value = btn.textContent;
  autoResize(input);
  input.focus();
}

function resetChat() {
  if (isRunning && !confirm("分析正在进行中，确定要开始新对话吗？")) return;
  isRunning = false;
  currentTaskId = null;
  $("#messages").innerHTML = "";
  $("#welcomeScreen").style.display = "flex";
  $("#workflowPanel").style.display = "none";
  $("#sendBtn").disabled = false;
  $("#queryInput").value = "";
  $("#queryInput").disabled = false;
  // Close sidebar on mobile
  $("#sidebar").classList.remove("open");
}

// ── Input Handling ───────────────────────────────────────────

function handleKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
}

// ── Core: Send & Stream ──────────────────────────────────────

async function sendQuery() {
  // 如果正在运行，点击按钮 = 终止
  if (isRunning) {
    cancelTask();
    return;
  }

  const input = $("#queryInput");
  const query = input.value.trim();
  if (!query) return;

  isRunning = true;
  $("#welcomeScreen").style.display = "none";
  // 发送按钮变成停止按钮
  const btn = $("#sendBtn");
  btn.disabled = false;
  btn.classList.add("stop-mode");
  btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>`;
  input.disabled = true;

  // Add user message
  addUserMessage(query);
  input.value = "";
  autoResize(input);

  // Show workflow panel
  $("#workflowPanel").style.display = "block";

  // Start the task
  try {
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      addSystemMessage("❌ " + (err.error || "启动失败"), "error");
      endRun();
      return;
    }

    const data = await resp.json();
    currentTaskId = data.task_id;

    // Add thinking indicator
    addThinking();

    // Start SSE stream
    connectSSE(currentTaskId);
  } catch (e) {
    addSystemMessage("❌ 网络错误: " + e.message, "error");
    endRun();
  }
}

function connectSSE(taskId) {
  const source = new EventSource(`/api/stream/${taskId}`);
  let currentLogContainer = null;
  let currentStepId = null;

  source.addEventListener("log", (e) => {
    const data = JSON.parse(e.data);

    if (data.type === "steps") {
      renderSteps(data.steps);
      return;
    }

    if (data.type === "step_start") {
      removeThinking();
      currentStepId = data.step_id;
      updateStepStatus(data.step_id, "active");
      currentLogContainer = addAgentMessage(data.step_id);
      addThinking();
      return;
    }

    if (data.type === "step_done") {
      updateStepStatus(data.step_id, "done");
      return;
    }

    if (data.type === "step_error") {
      updateStepStatus(data.step_id, "error");
      if (currentLogContainer) appendLog(currentLogContainer, "❌ " + data.text);
      return;
    }

    if (data.type === "status") {
      // Status messages from system
      if (!currentLogContainer) {
        currentLogContainer = addAgentMessage(data.agent || "system");
      }
      appendLog(currentLogContainer, data.text);
      return;
    }

    if (data.type === "error") {
      removeThinking();
      addSystemMessage(data.text, "error");
      return;
    }

    // Regular log line
    if (data.type === "log" && data.text) {
      if (currentLogContainer) {
        appendLog(currentLogContainer, data.text);
      }
    }
  });

  source.addEventListener("result", (e) => {
    const data = JSON.parse(e.data);
    removeThinking();

    if (data.final_report) {
      addReportMessage(data.final_report);
    }
  });

  source.addEventListener("done", (e) => {
    source.close();
    removeThinking();
    endRun();
  });

  source.onerror = () => {
    source.close();
    removeThinking();
    addSystemMessage("⚠️ 连接已断开", "warning");
    endRun();
  };
}

function endRun() {
  isRunning = false;
  // 恢复发送按钮
  const btn = $("#sendBtn");
  btn.disabled = false;
  btn.classList.remove("stop-mode");
  btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>`;
  $("#queryInput").disabled = false;
  $("#queryInput").focus();
}

async function cancelTask() {
  if (!currentTaskId) return;
  try {
    await fetch(`/api/cancel/${currentTaskId}`, { method: "POST" });
  } catch (e) {
    console.error("取消失败:", e);
  }
  // endRun 会在 SSE 收到 done/cancelled 事件后自动调用
}

// ── DOM: Messages ────────────────────────────────────────────

function addUserMessage(text) {
  const el = document.createElement("div");
  el.className = "message user";
  el.innerHTML = `<div class="msg-content">${escapeHtml(text)}</div>`;
  $("#messages").appendChild(el);
  scrollToBottom();
}

function addSystemMessage(text, type = "info") {
  const el = document.createElement("div");
  el.className = "message agent";
  el.innerHTML = `
    <div style="max-width:820px;margin:0 auto;">
      <div class="msg-agent-header">
        <div class="agent-avatar">⚙️</div>
        <span class="agent-name">系统</span>
        <span class="agent-time">${formatTime()}</span>
      </div>
      <div style="font-size:14px;color:${type === 'error' ? 'var(--error)' : 'var(--text-secondary)'};">
        ${escapeHtml(text)}
      </div>
    </div>`;
  $("#messages").appendChild(el);
  scrollToBottom();
}

function addAgentMessage(stepId) {
  const meta = AGENT_META[stepId] || { icon: "🤖", name: stepId };
  const el = document.createElement("div");
  el.className = "message agent";

  const logId = "log-" + stepId + "-" + Date.now();

  el.innerHTML = `
    <div style="max-width:820px;margin:0 auto;">
      <div class="msg-agent-header">
        <div class="agent-avatar">${meta.icon}</div>
        <span class="agent-name">${meta.name}</span>
        <span class="agent-time">${formatTime()}</span>
      </div>
      <div class="log-stream" id="${logId}"></div>
      <button class="toggle-log-btn" onclick="toggleLog(this, '${logId}')">
        ▼ 收起日志
      </button>
    </div>`;

  $("#messages").appendChild(el);
  scrollToBottom();

  return document.getElementById(logId);
}

function appendLog(container, text) {
  if (!container) return;

  // 检测图片标记 [IMAGE]filename[/IMAGE]
  const imgMatch = text.match(/\[IMAGE\](.+?)\[\/IMAGE\]/);
  if (imgMatch) {
    const filename = imgMatch[1];
    const wrapper = document.createElement("div");
    wrapper.className = "log-image";
    wrapper.innerHTML = `<img src="/outputs/${encodeURIComponent(filename)}" alt="${escapeHtml(filename)}" loading="lazy" onclick="window.open(this.src, '_blank')"/>`;
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
    scrollToBottom();
    return;
  }

  const line = document.createElement("div");
  line.className = "log-line";
  line.textContent = text;
  container.appendChild(line);

  container.scrollTop = container.scrollHeight;
  scrollToBottom();
}

function toggleLog(btn, logId) {
  const log = document.getElementById(logId);
  if (!log) return;
  const hidden = log.style.display === "none";
  log.style.display = hidden ? "block" : "none";
  btn.textContent = hidden ? "▼ 收起日志" : "▶ 展开日志";
}

function addReportMessage(markdown) {
  const el = document.createElement("div");
  el.className = "message agent";

  // Render markdown
  let html = "";
  try {
    html = marked.parse(markdown);
  } catch (e) {
    html = `<pre>${escapeHtml(markdown)}</pre>`;
  }

  el.innerHTML = `
    <div style="max-width:820px;margin:0 auto;">
      <div class="msg-agent-header">
        <div class="agent-avatar">📋</div>
        <span class="agent-name">最终报告</span>
        <span class="agent-time">${formatTime()}</span>
      </div>
      <div class="report-content">${html}</div>
    </div>`;

  $("#messages").appendChild(el);
  scrollToBottom();
}

function addThinking() {
  removeThinking(); // ensure only one
  const el = document.createElement("div");
  el.className = "thinking";
  el.id = "thinkingIndicator";
  el.innerHTML = `
    <div class="thinking-dots"><span></span><span></span><span></span></div>
    <span class="thinking-text">WORKING ON IT...</span>`;
  $("#messages").appendChild(el);
  scrollToBottom();
}

function removeThinking() {
  const el = document.getElementById("thinkingIndicator");
  if (el) el.remove();
}


// ── DOM: Workflow Steps ──────────────────────────────────────

function renderSteps(steps) {
  const list = $("#stepsList");
  list.innerHTML = "";

  steps.forEach((step, i) => {
    const el = document.createElement("div");
    el.className = "step-item";
    el.id = "step-" + step.id;
    el.innerHTML = `
      <div class="step-indicator">${i + 1}</div>
      <div class="step-info">
        <div class="step-label">${step.label}</div>
        <div class="step-desc">${step.desc}</div>
      </div>`;
    list.appendChild(el);
  });
}

function updateStepStatus(stepId, status) {
  // Remove active from all
  $$(".step-item.active").forEach(el => el.classList.remove("active"));

  const el = document.getElementById("step-" + stepId);
  if (!el) return;

  el.classList.remove("active", "done", "error");
  el.classList.add(status);

  if (status === "done") {
    const indicator = el.querySelector(".step-indicator");
    indicator.textContent = "✓";
  } else if (status === "error") {
    const indicator = el.querySelector(".step-indicator");
    indicator.textContent = "✗";
  }
}