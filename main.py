from db.neo4j.neo4j_connector import Neo4jEngine
from internal.langchain.wikipedia_api import WikipediaDocumentLoader

from dotenv import load_dotenv

load_dotenv()

from internal.llm.openai import ChatGptLLM


def main():
    page_title = "Marcus Aurelius"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "your_password"
    neo4j_engine = Neo4jEngine(neo4j_uri, neo4j_user, neo4j_password)

    # Step 1: Load and chunk Wikipedia page0
    wikipedia_loader = WikipediaDocumentLoader(page_title)
    chunks = wikipedia_loader.split_document(wikipedia_loader.load())

    # Step 2: Extract entities and relationships using OpenAI
    open_ai_llm = ChatGptLLM(chunks)
    relationships = open_ai_llm.generate_relationships_csv()

    # Step 3: Store entities and relationships in Neo4j
    neo4j_engine.store_in_neo4j_csv(relationships)
    neo4j_engine.store_named_relationships_from_file("relationships.csv")
    print(f"Successfully stored {len(relationships)} relationships in Neo4j!")


if __name__ == "__main__":
    main()
    main()
