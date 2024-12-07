import re
from transformers import pipeline
from internal.langchain.wikipedia_api import WikipediaDocumentLoader
from langchain.text_splitter import TokenTextSplitter
from neo4j import GraphDatabase

# Initialize BART model pipeline for text summarization
bart_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Neo4j configuration
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "your_password"


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
        summary = bart_pipeline(chunk, max_length=200, min_length=50, do_sample=False)
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
        """, entity1=relationship['entity1'], entity2=relationship['entity2'],
                    relationship_type=relationship['relationship_type'])

    session.close()


# Main Execution
if __name__ == "__main__":
    page_title = "Marcus Aurelius"

    # Step 1: Load and chunk Wikipedia page
    chunks = load_wikipedia_page(page_title)

    # Step 2: Use BART to generate summaries for each chunk
    summaries = generate_summary_using_bart(chunks)

    # Step 3: Extract entities and relationships from each summary
    all_entities = []
    all_relationships = []

    # Log entities and relationships
    for idx, summary in enumerate(summaries):
        print(f"\nSummary {idx + 1}: {summary}\n")

        entities = extract_entities(summary)
        print(f"Extracted Entities: {entities}")

        # Filter out common non-entity words (e.g., 'For', 'The', 'and')
        filtered_entities = [entity for entity in entities if
                             len(entity) > 1 and entity.lower() not in ["for", "the", "and", "in"]]
        print(f"Filtered Entities: {filtered_entities}")

        relationships = extract_relationships(filtered_entities)
        print(f"Extracted Relationships: {relationships}")

        all_entities.extend(filtered_entities)
        all_relationships.extend(relationships)

    # Step 4: Insert extracted entities and relationships into Neo4j
    insert_into_neo4j(all_entities, all_relationships)

    print("\nKnowledge graph has been successfully inserted into Neo4j.")
