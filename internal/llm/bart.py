import re
from dataclasses import dataclass, field
from transformers import pipeline, Pipeline


@dataclass
class BartLLM:
    chunks: list[str]
    bart_pipeline: Pipeline = field(init=False)

    def __post_init__(self):
        self.bart_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

    def generate_summary_using_bart(self):
        summaries = []
        for chunk in self.chunks:
            summary = self.bart_pipeline(chunk, max_length=200, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        return summaries

    @staticmethod
    def extract_entities(text):
        # Example of a simple regex-based entity extraction. You can use libraries like spaCy or other NER models.
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        return entities

    @staticmethod
    def extract_relationships(entities):
        relationships = []
        for i in range(len(entities) - 1):
            relationships.append({
                "entity1": entities[i],
                "entity2": entities[i + 1],
                "relationship_type": "RELATED"
            })
        return relationships
