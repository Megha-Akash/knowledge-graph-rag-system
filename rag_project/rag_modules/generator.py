"""
Answer Generator
Generates answers from retrieved KG context using Ollama.
Formats context as structured evidence triples for better grounding.
"""

import logging
import requests
from typing import List

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """\
You are a question answering system. Use the knowledge graph facts below to answer the question.
If the facts do not contain enough information, say "I don't know based on the provided context."

Knowledge Graph Facts:
{context}

Question: {question}

Answer:"""


class AnswerGenerator:
    """Generates answers using Ollama with structured KG context."""

    def __init__(self, model: str, endpoint: str,
                 temperature: float = 0.3, max_tokens: int = 512):
        self.model = model
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, question: str, context_strings: List[str]) -> str:
        """
        Generate answer from question and list of context strings.

        Args:
            question: Natural language question.
            context_strings: List of formatted KG path strings.

        Returns:
            Generated answer string.
        """
        if not context_strings:
            return "No relevant context found in knowledge graph."

        context = "\n".join(f"- {c}" for c in context_strings)
        prompt = ANSWER_PROMPT.format(context=context, question=question)
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error generating answer: {e}"