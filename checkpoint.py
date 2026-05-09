"""
Checkpoint 断点恢复 — 每个 agent 完成后自动保存 state，失败后可从断点恢复
"""

import os
import json
import time
import copy
from typing import Optional

import numpy as np
import pandas as pd

from config import OUTPUT_DIR


CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")


class CheckpointManager:
    """管理 state 的序列化、保存与加载"""

    def __init__(self, run_id: str = None):
        self.run_id = run_id or f"run_{int(time.time())}"
        self.run_dir = os.path.join(CHECKPOINT_DIR, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"📁 Checkpoint 目录: {self.run_dir}")

    def save(self, state: dict, step_name: str):
        """在某个 agent 完成后保存 state 快照"""
        filename = f"{step_name}.json"
        path = os.path.join(self.run_dir, filename)

        serializable = self._make_serializable(copy.deepcopy(state))

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)

        # 同时记录最新完成的步骤
        meta_path = os.path.join(self.run_dir, "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": self.run_id,
                "last_step": step_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "user_query": state.get("user_query", ""),
            }, f, ensure_ascii=False, indent=2)

        print(f"  💾 Checkpoint 已保存: {step_name}")

    def load(self, step_name: str) -> Optional[dict]:
        """加载指定步骤的 state 快照"""
        path = os.path.join(self.run_dir, f"{step_name}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_last_step(self) -> Optional[str]:
        """获取最近完成的步骤名"""
        meta_path = os.path.join(self.run_dir, "_meta.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("last_step")

    def load_latest(self) -> tuple:
        """加载最近的 checkpoint，返回 (step_name, state) 或 (None, None)"""
        step = self.get_last_step()
        if step is None:
            return None, None
        state = self.load(step)
        return step, state

    @staticmethod
    def list_runs() -> list:
        """列出所有可恢复的运行"""
        if not os.path.exists(CHECKPOINT_DIR):
            return []
        runs = []
        for d in sorted(os.listdir(CHECKPOINT_DIR), reverse=True):
            meta_path = os.path.join(CHECKPOINT_DIR, d, "_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                runs.append(meta)
        return runs

    def get_completed_steps(self) -> list:
        """返回所有已完成步骤名列表（根据目录中存在的 json 文件判断）"""
        if not os.path.exists(self.run_dir):
            return []
        steps = []
        for f in os.listdir(self.run_dir):
            if f.endswith(".json") and f != "_meta.json":
                steps.append(f[:-5])
        return steps

    @classmethod
    def find_latest_run(cls, query: str):
        """查找同一 query 的最新运行，返回 CheckpointManager 或 None"""
        if not os.path.exists(CHECKPOINT_DIR):
            return None
        for d in sorted(os.listdir(CHECKPOINT_DIR), reverse=True):
            meta_path = os.path.join(CHECKPOINT_DIR, d, "_meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("user_query") == query:
                return cls(run_id=d)
        return None

    def _make_serializable(self, obj):
        """递归清理不可序列化的对象"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        else:
            return obj