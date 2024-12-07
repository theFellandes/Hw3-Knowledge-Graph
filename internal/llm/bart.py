from dataclasses import dataclass, field
from transformers import pipeline, Pipeline

from internal.llm.llm import LLMBase


@dataclass
class BartLLM(LLMBase):
    chunks: list[str]
    _bart_pipeline: Pipeline = field(init=False)

    def __post_init__(self):
        self._bart_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

    def generate_summary(self):
        summaries = []
        for chunk in self.chunks:
            summary = self._bart_pipeline(chunk, max_length=200, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        return summaries
