from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
from uuid import uuid4
from datetime import datetime
import traceback, sys

from .agents import ExecutionAgent, ContextAgent, TaskCreationAgent, PrioritizationAgent
from .memory import VectorMemory

def safe_print(prefix: str, value: Any):
    try:
        text = value if isinstance(value, str) else repr(value)
        print(prefix + ("\n" if prefix and not prefix.endswith("\n") else "") + text + ("\n" if not text.endswith("\n") else ""), flush=True)
    except Exception as e:
        print(f"[print-error] {e}", flush=True)

@dataclass
class BabyAGILoop:
    objective: str
    initial_tasks: List[str]
    exec_agent: ExecutionAgent
    ctx_agent: ContextAgent
    create_agent: TaskCreationAgent
    prio_agent: PrioritizationAgent
    memory: VectorMemory
    max_steps: int = 6
    top_k_context: int = 5
    temperature: float = 0.2

    task_queue: List[str] = field(default_factory=list)
    done_tasks: List[str] = field(default_factory=list)

    def _build_context(self) -> str:
        # top_k_context가 0 이하이면 검색하지 않고 빈 문자열 반환
        if self.top_k_context <= 0:
            return ""
        hits = self.memory.search(self.objective, k=self.top_k_context)
        if not hits:
            return ""
        bullets = []
        for h in hits:
            meta = h.get("meta") or {}
            src = f"(step={meta.get('step')}, task='{(meta.get('task') or '')[:40]}...')"
            bullets.append(f"- {h['text']}\n  {src}")
        return "\n".join(bullets)


    def run(self) -> List[Dict[str, Any]]:
        logs: List[Dict[str, Any]] = []
        self.task_queue = list(self.initial_tasks)

        for step in range(1, self.max_steps + 1):
            try:
                if not self.task_queue:
                    print(f"[종료] 대기열이 비어 종료 (단계 {step}).", flush=True)
                    break

                task = self.task_queue.pop(0)
                context = self._build_context()
                print(f"\n=== 단계 {step} ===", flush=True)
                print(f"[작업] {task}", flush=True)
                if context:
                    print(f"[컨텍스트] {len(context.splitlines())}줄 검색됨", flush=True)

                # 1) 실행
                result = self.exec_agent.run(self.objective, task, context, temperature=self.temperature)
                safe_print("[결과]", result)

                # 2) 요약 + 메모리 저장
                nugget = self.ctx_agent.summarize(self.objective, result)
                # self.memory.add(
                #     doc_id=str(uuid4()),
                #     text=nugget,
                #     metadata={"step": step, "task": task, "ts": datetime.utcnow().isoformat()},
                # )

                # 3) 새 후보 생성
                candidates = self.create_agent.propose(self.objective, task, result, self._build_context())
                print("[후보 과제]", flush=True)
                for i, c in enumerate(candidates, 1):
                    print(f"  {i}. {c}", flush=True)

                # 4) 우선순위
                prioritized = self.prio_agent.prioritize(self.objective, candidates, self.done_tasks)
                print("[우선 과제(정렬 결과)]", flush=True)
                for i, p in enumerate(prioritized, 1):
                    print(f"  {i}. {p}", flush=True)

                # 5) 큐 업데이트
                for t in prioritized:
                    if t not in self.task_queue:
                        self.task_queue.append(t)

                self.done_tasks.append(task)
                logs.append({
                    "step": step,
                    "task": task,
                    "result": result,
                    "summary": nugget,
                    "candidates": candidates,
                    "prioritized": prioritized,
                })
            except Exception as e:
                print("[ERROR] Exception in loop step:", e, flush=True)
                traceback.print_exc()
                break

        print("\n[종료] 완료 또는 최대 단계 도달.", flush=True)
        return logs
