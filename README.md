# BabyAGI Mini (VS Code Script Edition, Korean Prompts)

로컬 벡터 메모리(ChromaDB) + OpenAI(Chat/Embedding)로 구성한 교육용 BabyAGI 미니 구현입니다.
프롬프트 및 실행 출력은 한국어 중심으로 구성되어 있습니다.

## 1) 설치
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) 환경변수
`.env`를 만들고 OpenAI 키를 넣어주세요.
```env
OPENAI_API_KEY=sk-...
```

## 3) 실행
```bash
python run.py   --objective "Tree-of-Thoughts(ToT)의 개념, 탐색 전략(DFS/Beam 등), 구현 시 유의점 1가지를 담아 간결한 학습 브리프를 작성한다."
# 또는 기본값 실행
python run.py
```

## 4) 메모리 초기화
ChromaDB 저장 경로 기본값은 `.babyagi_memory` 입니다. 초기화하려면 폴더 삭제:
```bash
rm -rf .babyagi_memory
```
