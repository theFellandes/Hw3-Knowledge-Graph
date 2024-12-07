from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import WikipediaLoader
from neo4j import GraphDatabase


# Connect to Neo4j
uri = "bolt://localhost:7687"  # Update with your Neo4j URI
driver = GraphDatabase.driver(uri, auth=("neo4j", "your_password"))

def store_in_neo4j(relationships: list):
    with driver.session() as session:
        for rel in relationships:
            query = """
            MERGE (a:Entity {name: $source})
            MERGE (b:Entity {name: $target})
            MERGE (a)-[r:RELATIONSHIP {type: $relationship}]->(b)
            """
            session.run(query, source=rel['source'], target=rel['target'], relationship=rel['relationship'])

##################################################

# Load the Wikipedia page for Marcus Aurelius
def load_wikipedia_page(page_title: str):
    loader = WikipediaLoader(page_title)
    documents = loader.load()
    return documents[0].page_content  # Extract the content

# Example: Load content for Marcus Aurelius
page_content = load_wikipedia_page("Marcus Aurelius")


##################################################

# Extract relationships
def extract_relationships(text: str) -> list:
    response = entity_extraction_chain.run(text=text)
    return eval(response.strip())  # Replace with JSON parsing for production

##################################################


# OpenAI LLM setup
llm = OpenAI(model="text-davinci-003", temperature=0, max_tokens=500)

# Define a prompt for extracting entities and relationships
prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Analyze the following text and extract entities (e.g., person, place, event) and their relationships 
    in a knowledge graph format. Provide the result as a JSON array of objects. Each object should have 
    "source", "relationship", and "target" keys. Ensure the JSON is well-formatted.

    Text: {text}

    Example output:
    [
        {"source": "Marcus Aurelius", "relationship": "wrote", "target": "Meditations"},
        {"source": "Marcus Aurelius", "relationship": "was born in", "target": "Rome"}
    ]
    """
)

# Create an LLMChain
entity_extraction_chain = LLMChain(llm=llm, prompt=prompt)

print(f"{entity_extraction_chain=}")

# Example: Extract entities and relationships
relationships = extract_relationships(page_content[:1000])  # Use a smaller chunk

print(f"{relationships=}")

# Example: Store relationships in Neo4j
store_in_neo4j(relationships)
