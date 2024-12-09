import re

from dataclasses import dataclass
from abc import ABC, abstractmethod
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache


@dataclass
class LLMBase(ABC):
    chunks: list[str]

    @staticmethod
    def enable_cache():
        set_llm_cache(InMemoryCache())

    @abstractmethod
    def generate_summary(self):
        raise NotImplementedError

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
