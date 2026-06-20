"""
DataAnalystAgent class and interactive CLI.
Main orchestration logic for the WDI analysis agent.
"""

from typing import Any
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.agent import ReActAgent

from core.store import WDIStore
from tools import (
    create_inspect_schema_tool,
    create_retrieve_docs_tool,
    create_retrieve_metadata_tool,
    create_run_python_tool,
)

from .config import OPENROUTER_API_KEY, OPENROUTER_MODEL
from .prompts import SYSTEM_PROMPT


class DataAnalystAgent:
    def __init__(self):
        self.store = WDIStore()
        self.llm = OpenRouter(
            api_key=OPENROUTER_API_KEY, max_tokens=1024, context_window=4096, model=OPENROUTER_MODEL
        )

        self.tools = [
            create_inspect_schema_tool(self.store),
            create_retrieve_docs_tool(self.store),
            create_retrieve_metadata_tool(self.store),
            create_run_python_tool(self.store.df),
        ]

        self.agent = ReActAgent(
            name="DataAnalystAgent",
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            max_iterations=8,
            system_prompt=SYSTEM_PROMPT,
        )

    async def aquery(self, question: str) -> str:
        try:
            response = await self.agent.run(
                user_msg=question, max_iterations=8, early_stopping_method="generate"
            )
            answer = str(response)
            answer = self._remove_tool_suggestions(answer)
            return answer
        except Exception as e:
            msg = str(e)
            if "Max iterations" in msg or "max_iterations" in msg:
                response = await self.agent.run(user_msg=question, early_stopping_method="generate")
                answer = str(response)
                answer = self._remove_tool_suggestions(answer)
                return answer
            raise

    def _remove_tool_suggestions(self, text: str) -> str:
        """Remove suggestions like 'you can use the run_python() tool' or 'please use...'"""
        lines = text.split('\n')
        filtered = []
        for line in lines:
            lower = line.lower()
            if any(pattern in lower for pattern in [
                'you can use',
                'please use',
                'use the run_python()',
                'use the retrieve_',
                'use the inspect_',
                'for the exact figure',
                'for exact data',
                'please run',
                'you can run',
            ]):
                if not any(word in lower for word in ['found', 'retrieved', 'got', 'shows', 'indicates']):
                    continue
            filtered.append(line)
        return '\n'.join(filtered).strip()

    def query(self, question: str, timeout: int = 30) -> str:
        import asyncio
        return asyncio.run(self.aquery(question))

    def answer(self, question: str, timeout: int = 30) -> str:
        return self.query(question, timeout=timeout)

    def format_answer(self, answer: Any) -> str:
        if isinstance(answer, str):
            return answer
        try:
            return str(answer)
        except Exception:
            import json
            return json.dumps(answer, ensure_ascii=False)


def interactive():
    agent = DataAnalystAgent()
    print("Data Analyst Agent — RAG semantico (sentence-transformers + FAISS). Type 'exit' to quit.")
    while True:
        q = input("Question> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        print("Thinking... (this may take a few seconds)")
        try:
            print(agent.query(q))
        except Exception as e:
            print(f"Agent error: {e}")


if __name__ == "__main__":
    interactive()
