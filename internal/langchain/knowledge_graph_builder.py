from dataclasses import field, dataclass

from langchain_core.language_models import BaseLanguageModel
from langchain_experimental.graph_transformers import LLMGraphTransformer
from typing_extensions import deprecated


@dataclass
class KnowledgeGraphBuilder:
    """Class to process data into a knowledge graph."""
    llm: BaseLanguageModel
    llm_transformer: LLMGraphTransformer = field(init=False)

    def __post_init__(self):
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)

    def convert(self, documents):
        return self.llm_transformer.convert_to_graph_documents(documents)

    @staticmethod
    def process_and_store(output, db_connection):
        lines = output.strip().split("\n")
        for line in lines:
            if "->" in line:  # Indicates a relationship
                source, rest = line.split("->")
                target, relationship = rest.split("[")
                target = target.strip()
                relationship = relationship.strip("]")
                db_connection.create_relationship(source.strip(), target.strip(), relationship.strip())
            else:  # Indicates a node
                node_type, node_name = line.split(":")
                node_type = node_type.strip()
                node_name = node_name.strip()
                db_connection.create_node(node_type, {"name": node_name})
