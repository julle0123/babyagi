from __future__ import annotations
import argparse, json, os, sys, traceback
from dotenv import load_dotenv

from app.config import load_settings
from app.llm import LLMClient
from app.memory import VectorMemory
from app.agents import ExecutionAgent, ContextAgent, TaskCreationAgent, PrioritizationAgent
from app.loop import BabyAGILoop

def main():
    try:
        print("[DEBUG] start run.py", flush=True)
        load_dotenv()
        print("[DEBUG] .env loaded", flush=True)

        cfg = load_settings()
        print("[DEBUG] settings loaded:", cfg, flush=True)

        llm = LLMClient(cfg.openai_api_key, cfg.chat_model, cfg.embed_model)
        print("[DEBUG] LLM client ready", flush=True)
        memory = VectorMemory(cfg.persist_dir, embed_fn=llm.embed)
        print("[DEBUG] Vector memory ready at", cfg.persist_dir, flush=True)

        exec_agent = ExecutionAgent(llm)
        ctx_agent = ContextAgent(llm)
        create_agent = TaskCreationAgent(llm)
        prio_agent = PrioritizationAgent(llm)

        parser = argparse.ArgumentParser(description="Mini BabyAGI Runner (Korean prompts, debug)")
        parser.add_argument("--objective", type=str, default="Tree-of-Thoughts(ToT)의 개념, 탐색 전략(DFS/Beam 등), 구현 시 유의점 1가지를 담아 간결한 학습 브리프를 작성한다.")
        parser.add_argument("--steps", type=int, default=cfg.max_steps)
        parser.add_argument("--topk", type=int, default=cfg.top_k)
        parser.add_argument("--temp", type=float, default=cfg.temperature)
        args = parser.parse_args()

        initial_tasks = [
            "ToT 학습 브리프에 반드시 포함되어야 할 핵심 항목을 4~6개로 개요화한다.",
            "일반인이 이해할 수 있도록 ToT와 CoT의 차이를 200단어 내외로 설명 초안을 작성한다.",
        ]

        loop = BabyAGILoop(
            objective=args.objective,
            initial_tasks=initial_tasks,
            exec_agent=exec_agent,
            ctx_agent=ctx_agent,
            create_agent=create_agent,
            prio_agent=prio_agent,
            memory=memory,
            max_steps=args.steps,
            top_k_context=args.topk,
            temperature=args.temp,
        )

        logs = loop.run()
        print(f"\n[DEBUG] total steps: {len(logs)}", flush=True)

        # 옵션: 로그 파일로 저장
        try:
            with open("run_logs.jsonl", "w", encoding="utf-8") as f:
                for row in logs:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("[DEBUG] 로그 저장: run_logs.jsonl", flush=True)
        except Exception as e:
            print("[WARN] 로그 저장 실패:", e, flush=True)

    except Exception as e:
        print("[FATAL] Uncaught exception:", e, flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
