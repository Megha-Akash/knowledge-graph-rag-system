"""
Prompt Templates for KG Extraction
Returns entities with label + properties dict, relations with start_entity/end_entity/label.
"""

EXTRACTION_PROMPT = """\
You are a knowledge graph extraction system. Extract entities, relations, and topics from the text below.

Return ONLY a valid JSON object. No explanation, no markdown, no extra text.

Format:
{{
  "entities": [
    {{"label": "<type>", "properties": {{"name": "<name>", ...}}}}
  ],
  "relations": [
    {{"start_entity": {{"name": "<name>"}}, "end_entity": {{"name": "<name>"}}, "label": "<RELATION_TYPE>"}}
  ],
  "topics": ["<topic1>", "<topic2>"]
}}

Rules:
- Entity label must be one of: Person, Organization, Location, Movie, Book, Event, Work, Award, Date, Other
- Relation label must be UPPER_SNAKE_CASE e.g. DIRECTED, ACTED_IN, FOUNDED, LOCATED_IN, PART_OF
- Only extract entities clearly mentioned in the text
- Only extract relations explicitly stated, not implied
- topics should be 2-5 general themes from the text
- If no entities found return {{"entities": [], "relations": [], "topics": []}}

Title: {title}
Text: {text}

JSON:"""
