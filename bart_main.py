from langchain_text_splitters import TokenTextSplitter

from internal.llm.bart import BartLLM
from internal.langchain.wikipedia_api import WikipediaDocumentLoader
from db.neo4j.neo4j_connector import Neo4jEngine


def load_wikipedia_page(page_title: str):
    loader = WikipediaDocumentLoader(page_title)
    raw_documents = loader.load()
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = splitter.split_documents(raw_documents)
    return [doc.page_content for doc in documents]


def main():
    page_title = "Marcus Aurelius"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "your_password"
    neo4j_engine = Neo4jEngine(neo4j_uri, neo4j_user, neo4j_password)

    # Step 1: Load and chunk Wikipedia page
    chunks = load_wikipedia_page(page_title)
    llm = BartLLM(chunks)

    # Step 2: Use BART to generate summaries for each chunk
    summaries = llm.generate_summary_using_bart()

    # Step 3: Extract entities and relationships from each summary
    all_entities = []
    all_relationships = []

    # Log entities and relationships
    for idx, summary in enumerate(summaries):
        print(f"\nSummary {idx + 1}: {summary}\n")

        entities = llm.extract_entities(summary)
        print(f"Extracted Entities: {entities}")

        # Filter out common non-entity words (e.g., 'For', 'The', 'and')
        filtered_entities = [entity for entity in entities if
                             len(entity) > 1 and entity.lower() not in ["for", "the", "and", "in"]]
        print(f"Filtered Entities: {filtered_entities}")

        relationships = llm.extract_relationships(filtered_entities)
        print(f"Extracted Relationships: {relationships}")

        all_entities.extend(filtered_entities)
        all_relationships.extend(relationships)

    # Step 4: Insert extracted entities and relationships into Neo4j
    neo4j_engine.insert_into_neo4j(all_entities, all_relationships)

    print("\nKnowledge graph has been successfully inserted into Neo4j.")


if __name__ == "__main__":
    main()
