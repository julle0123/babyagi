from __future__ import annotations
from dataclasses import dataclass
from typing import List

from .llm import LLMClient

def _parse_numbered_list(raw: str, limit: int = 6) -> List[str]:
    items: List[str] = []
    for line in (raw or "").splitlines():
        s = line.strip()
        if len(s) >= 3 and s[0].isdigit() and s[1] in ".)":
            items.append(s.split(" ", 1)[1] if " " in s else s)
    return items[:limit]

@dataclass
class ExecutionAgent:
    llm: LLMClient
    def run(self, objective: str, task: str, context: str, temperature: float, max_tokens: int = 800) -> str:
        system = "당신은 정확하고 간결하며 구조화된 출력을 생성하는 전문가 어시스턴트입니다. 가능하다면 근거(출처/근거 논리)를 함께 제시하세요."
        user = f"""[목표(Objective)]
{objective}

[작업(Task)]
{task}

[컨텍스트(과거 유용한 결과 요약)]
{context if context else "(컨텍스트 없음)"}

[지시사항]
- 위 목표 달성을 위해, 제시된 '작업'만 해결하세요.
- 실행 가능한 구체적 출력으로 답하세요. 꼭 필요하지 않다면 300자(또는 300단어) 이내로 간결하게.
- 추정/가정이 필요하면 합리적 가정을 밝히고 진행하세요.
"""
        return self.llm.chat(system, user, temperature=temperature, max_tokens=max_tokens) or ""

@dataclass
class ContextAgent:
    llm: LLMClient
    def summarize(self, objective: str, last_result: str) -> str:
        system = "당신은 요약 전문가입니다."
        user = f"""다음 결과를 이후 작업에 재사용하기 좋은 '핵심 지식 덩어리'로 요약하세요.

[목표]
{objective}

[요약 대상 결과]
{last_result}

[규칙]
- 글머리표 4~6개.
- 사실 중심, 중복/군더더기 제거.
- 핵심 데이터/제약/결정 사항을 포함.
"""
        return self.llm.chat(system, user, temperature=0.1, max_tokens=300) or ""

@dataclass
class TaskCreationAgent:
    llm: LLMClient
    def propose(self, objective: str, last_task: str, last_result: str, known_context: str) -> List[str]:
        system = "당신은 목표 달성에 필요한 '작고 검증 가능한 과제'를 생성하는 에이전트입니다."
        user = f"""[목표]
{objective}

[직전 작업]
{last_task}

[직전 작업 결과]
{last_result}

[알려진 컨텍스트(요약)]
{known_context}

[지시사항]
- 논리적으로 이어지는 '새 과제' 3~6개를 제안.
- 과제는 작고(단일 목적), 검증 가능하며, 구체적 산출물이 있어야 함.
- 중복/모호/지나치게 포괄적인 과제 금지.
- 출력 형식: 번호 목록으로만 (예: "1. ...", "2. ...").
"""
        raw = self.llm.chat(system, user, temperature=0.2, max_tokens=400)
        return _parse_numbered_list(raw, limit=6)

@dataclass
class PrioritizationAgent:
    llm: LLMClient
    def prioritize(self, objective: str, candidates: List[str], done_tasks: List[str]) -> List[str]:
        system = "당신은 엄격한 우선순위 결정 엔진입니다."
        user = f"""[목표]
{objective}

[후보 과제]
{chr(10).join(f"- {t}" for t in candidates)}

[이미 완료된 과제]
{chr(10).join(f"- {t}" for t in done_tasks) if done_tasks else "(없음)"}

[규칙]
- 영향도/실현가능성/신규성 기준으로 상위 3~5개만 선정해 순서대로 나열.
- 중복 및 완료된 과제 제거.
- 과제는 원자적(atomic)이고 검증 가능해야 함.
- 출력 형식: 번호 목록으로만 (예: "1. ...", "2. ...").
"""
        raw = self.llm.chat(system, user, temperature=0.1, max_tokens=300)
        return _parse_numbered_list(raw, limit=5)
