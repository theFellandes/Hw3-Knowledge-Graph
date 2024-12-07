from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFaceEndpoint
import yaml
from neo4j import GraphDatabase
from typing import Dict


class YamlReader:
    """Class to read configuration from a YAML file."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.config = None

    def __enter__(self):
        with open(self.file_path, 'r') as file:
            self.config = yaml.safe_load(file)
        return self.config

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class Neo4jConnection:
    """Class to manage Neo4j connections and transactions."""

    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def __enter__(self):
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.driver:
            self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            return session.run(query, parameters or {})

    def create_node(self, node_type, properties):
        query = f"""
        MERGE (n:{node_type} {{name: $name}})
        SET n += $props
        RETURN n
        """
        self.run_query(query, {"name": properties["name"], "props": properties})

    def create_relationship(self, source, target, relationship):
        query = f"""
        MATCH (a {{name: $source}}), (b {{name: $target}})
        MERGE (a)-[r:{relationship}]->(b)
        RETURN r
        """
        self.run_query(query, {"source": source, "target": target})


class WikipediaParser:
    """Class to retrieve Wikipedia content using LangChain's WikipediaAPIWrapper."""

    def __init__(self):
        self.wiki = WikipediaAPIWrapper()

    def fetch_content(self, page_title):
        return self.wiki.run(page_title)


class KnowledgeGraphBuilder:
    """Class to process data into a knowledge graph format."""

    def __init__(self, llm, prompt_template):
        # Combine prompt and LLM using RunnableSequence
        self.chain = prompt_template | llm

    def parse_text(self, text: str) -> str:
        """Parse the input text into a knowledge graph format."""
        try:
            result = self.chain.invoke({"text": text})
            if result is None:
                print("Error: Chain returned None.")
                return None
            print("Output from chain.invoke:", result.strip())
            return result.strip()
        except Exception as e:
            print("Error during chain.invoke execution:", e)
            return None

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


if __name__ == "__main__":
    # Load configuration from YAML file
    CONFIG_PATH = "../internal/reader/config.yaml"

    try:
        with YamlReader(CONFIG_PATH) as config:
            neo4j_config = config["neo4j"]
            wikipedia_config = config["wikipedia"]
            huggingface_config = config["huggingface"]

            # Neo4j connection details
            NEO4J_URI = neo4j_config["uri"]
            NEO4J_USER = neo4j_config["user"]
            NEO4J_PASSWORD = neo4j_config["password"]

            # LangChain components
            llm = HuggingFaceEndpoint(
                model=huggingface_config["model"],  # Free Hugging Face model
                temperature=huggingface_config["temperature"],
                huggingfacehub_api_token=huggingface_config["token"],
                max_new_tokens=huggingface_config["max_new_tokens"],
            )

            prompt_template = PromptTemplate(
                template="""
                You are an expert knowledge graph builder. Parse the following text into a set of nodes and relationships for a knowledge graph.
                Each node should have a type and name. Format:
                EntityType: EntityName
                EntityName1 -> EntityName2 [RelationshipType]

                Input text:
                {text}
                """,
                input_variables=["text"]
            )

            # Set Wikipedia page title for Al-Khwarizmi
            page_title = "Al-Khwarizmi"

            # Step 1: Fetch Wikipedia content directly using WikipediaAPIWrapper
            wiki_parser = WikipediaParser()
            wiki_content = wiki_parser.fetch_content(page_title)

            # Step 2: Parse content into a knowledge graph
            kg_builder = KnowledgeGraphBuilder(llm=llm, prompt_template=prompt_template)
            parsed_output = kg_builder.parse_text(wiki_content)

            print("Parsed Output:\n", parsed_output)  # Debugging purpose

            # Step 3: Store knowledge graph into Neo4j
            with Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as neo4j_conn:
                kg_builder.process_and_store(parsed_output, neo4j_conn)

            print("Knowledge graph successfully created in Neo4j!")

    except Exception as e:
        print(f"An error occurred: {e}")
