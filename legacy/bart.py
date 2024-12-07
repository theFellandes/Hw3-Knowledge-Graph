from transformers import pipeline
from internal.langchain.wikipedia_api import WikipediaDocumentLoader
from langchain.text_splitter import TokenTextSplitter
from neo4j import GraphDatabase
from typing import List
import re

# Load BART model
generator = pipeline("summarization", model="facebook/bart-large-cnn")

# Neo4j configuration
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "your_password"


# Function to load Wikipedia page
# Load Wikipedia page and split it into chunks
def load_wikipedia_page(page_title: str):
    loader = WikipediaDocumentLoader(page_title)
    raw_documents = loader.load()
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = splitter.split_documents(raw_documents)
    return [doc.page_content for doc in documents]


# Use BART to generate a summary for each chunk
def generate_summary_using_bart(chunks):
    summaries = []
    for chunk in chunks:
        summary = generator(chunk, max_length=200, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return summaries


# Extract entities from text using simple NER pattern (can be replaced with more advanced methods)
def extract_entities(text):
    # Example of a simple regex-based entity extraction. You can use libraries like spaCy or other NER models.
    entities = re.findall(r'\b[A-Z][a-z]+\b', text)
    return entities

# Extract relationships from entities (very simple for this example)
def extract_relationships(entities):
    relationships = []
    for i in range(len(entities) - 1):
        relationships.append({
            "entity1": entities[i],
            "entity2": entities[i + 1],
            "relationship_type": "RELATED"
        })
    return relationships


# Function to extract knowledge graph using BART
def extract_knowledge_graph_from_chunks(chunks: List[str]) -> List[str]:
    knowledge_graph = []
    for chunk in chunks:
        prompt = f"""
        Parse the text below into a knowledge graph with detailed nodes and relationships.
        Include entity types and relationships between them. Use the following format:

        - EntityType: EntityName
        - EntityName1 -> EntityName2 [RelationshipType]

        Input Text:
        {chunk}
        """
        response = generator(prompt, max_length=512, truncation=True)
        knowledge_graph.append(response[0]['generated_text'])
    return knowledge_graph


# Function to clean and parse knowledge graph output
def parse_knowledge_graph(knowledge_graph: List[str]) -> List[dict]:
    parsed_data = []
    for text in knowledge_graph:
        lines = text.split('\n')
        for line in lines:
            # Match entities and relationships
            entity_match = re.match(r"([A-Za-z\s]+):\s([A-Za-z\s]+)", line)
            relationship_match = re.match(r"([A-Za-z\s]+)\s->\s([A-Za-z\s]+)\s\[(.*)\]", line)
            if entity_match:
                entity_type, entity_name = entity_match.groups()
                parsed_data.append({"type": entity_type, "name": entity_name.strip()})
            elif relationship_match:
                entity1, entity2, relationship_type = relationship_match.groups()
                parsed_data.append({
                    "entity1": entity1.strip(),
                    "entity2": entity2.strip(),
                    "relationship_type": relationship_type.strip()
                })
    print(f"Parsed {len(knowledge_graph)} knowledge graph")
    return parsed_data


# Function to insert parsed knowledge graph into Neo4j
# Insert entities and relationships into Neo4j
def insert_into_neo4j(entities, relationships):
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    session = driver.session()

    for entity in entities:
        # Insert entity nodes into Neo4j
        session.run("""
        MERGE (e:Entity {name: $name})
        """, name=entity)

    for relationship in relationships:
        # Insert relationship into Neo4j
        session.run("""
        MATCH (a:Entity {name: $entity1}), (b:Entity {name: $entity2})
        MERGE (a)-[:`$relationship_type`]->(b)
        """, entity1=relationship['entity1'], entity2=relationship['entity2'], relationship_type=relationship['relationship_type'])

    session.close()


# Main Execution
if __name__ == "__main__":
    page_title = "Marcus Aurelius"

    # Step 1: Load and chunk Wikipedia page
    chunks = load_wikipedia_page(page_title)

    # Step 2: Use BART to generate summaries for each chunk
    summaries = generate_summary_using_bart(chunks)

    # Step 3: Extract entities from each summary
    all_entities = []
    all_relationships = []
    for summary in summaries:
        entities = extract_entities(summary)
        relationships = extract_relationships(entities)

        all_entities.extend(entities)
        all_relationships.extend(relationships)

    # Step 4: Insert extracted entities and relationships into Neo4j
    insert_into_neo4j(all_entities, all_relationships)

    print("Knowledge graph has been successfully inserted into Neo4j.")