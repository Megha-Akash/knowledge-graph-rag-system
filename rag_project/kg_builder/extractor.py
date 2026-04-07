"""
KG Extractor
Calls Ollama LLM and parses entity/relation JSON from the response.
Uses the same extraction format as the FZI version (entities with properties dict,
relations with start_entity/end_entity/label).
"""

import json
import re
import logging
import requests
from typing import Dict, Any, Optional

from .prompt_templates import EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class KGExtractor:
    """
    Extracts entities and relations from a document using an Ollama-served LLM.
    """

    def __init__(self, model: str, endpoint: str, temperature: float = 0.1, max_tokens: int = 1500):
        self.model = model
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(self, doc: dict) -> Dict[str, Any]:
        """
        Extract KG triples from a single document.

        Args:
            doc: Dict with keys: idx, title, text.

        Returns:
            Dict with keys: entities, relations, topics.
            Returns empty lists on failure — never raises.
        """
        prompt = EXTRACTION_PROMPT.format(
            title=doc.get("title", ""),
            text=doc.get("text", ""),
        )

        raw = self._call_ollama(prompt)
        if raw is None:
            logger.warning(f"[idx={doc.get('idx')}] LLM call failed, returning empty extraction")
            return {"entities": [], "relations": [], "topics": []}

        return self._parse_and_validate(raw, doc.get("idx"))

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """POST to Ollama generate endpoint, return raw text or None on error."""
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
            return response.json().get("response", "")
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
        return None

    def _parse_and_validate(self, raw: str, idx: Any) -> dict:
        """Parse JSON and validate structure matches FZI format."""
        data = self._parse_json(raw, idx)

        entities = []
        for e in data.get("entities", []):
            if isinstance(e, dict) and "label" in e and "properties" in e:
                if isinstance(e["properties"], dict) and e["properties"].get("name"):
                    entities.append(e)

        relations = []
        for r in data.get("relations", []):
            if (isinstance(r, dict)
                    and isinstance(r.get("start_entity"), dict)
                    and isinstance(r.get("end_entity"), dict)
                    and "label" in r
                    and r["start_entity"].get("name")
                    and r["end_entity"].get("name")):
                relations.append(r)

        topics = []
        for t in data.get("topics", []):
            if isinstance(t, str) and t.strip():
                topics.append(t.strip())

        return {"entities": entities, "relations": relations, "topics": topics}

    def _parse_json(self, raw: str, idx: Any) -> dict:
        """Multi-strategy JSON parser."""
        # Strategy 1: direct parse
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract first {...} block
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 3: strip markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        logger.warning(f"[idx={idx}] All JSON parse strategies failed. Raw snippet: {raw[:200]}")
        return {}