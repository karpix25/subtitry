from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import time
from typing import Dict, Optional
from uuid import uuid4


@dataclass
class TaskRecord:
    task_id: str
    status: str = "pending"
    result: Optional[dict] = None
    error: Optional[str] = None
    callback_url: Optional[str] = None
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "callback_url": self.callback_url,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TaskManager:
    def __init__(self) -> None:
        self._tasks: Dict[str, TaskRecord] = {}
        self._lock = Lock()

    def create_task(self, callback_url: Optional[str] = None) -> str:
        task_id = uuid4().hex
        with self._lock:
            self._tasks[task_id] = TaskRecord(task_id=task_id, callback_url=callback_url)
        return task_id

    def mark_processing(self, task_id: str) -> None:
        self._update(task_id, status="processing")

    def mark_completed(self, task_id: str, result: dict) -> None:
        self._update(task_id, status="completed", result=result, error=None)

    def mark_failed(self, task_id: str, error: str) -> None:
        self._update(task_id, status="failed", error=error)

    def get(self, task_id: str) -> Optional[dict]:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            data = record.to_dict()
            if isinstance(data.get("result"), dict):
                data["result"] = data["result"].copy()
            return data

    def _update(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                raise KeyError(f"Unknown task_id {task_id}")
            if status is not None:
                record.status = status
            if result is not None:
                record.result = result
            if error is not None:
                record.error = error
            record.updated_at = time()
