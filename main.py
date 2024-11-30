import getpass
import os

from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

from db.neo4j.neo4j_connector import Neo4jEngine
from internal.reader.yaml_reader import YamlReader
from internal.langchain.wikipedia_api import WikipediaDocumentLoader
from internal.langchain.knowledge_graph_builder import KnowledgeGraphBuilder

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


def main():
    CONFIG_PATH = "internal/reader/config.yaml"

    try:
        with YamlReader(CONFIG_PATH) as config:
            neo4j_config = config["neo4j"]
            wikipedia_config = config["wikipedia"]
            huggingface_config = config["huggingface"]

            # Neo4j connection details
            NEO4J_URI = neo4j_config["uri"]
            NEO4J_USER = neo4j_config["user"]
            NEO4J_PASSWORD = neo4j_config["password"]

    except Exception as e:
        print(f"An error occurred: {e}")


    # LangChain components
    llm = HuggingFaceEndpoint(
        model=huggingface_config["model"],  # Free Hugging Face model
        temperature=huggingface_config["temperature"],
        huggingfacehub_api_token=huggingface_config["token"],
        max_new_tokens=huggingface_config["max_new_tokens"],
    )

    # os.environ["OPENAI_API_KEY"] = getpass.getpass()
    # llm = ChatOpenAI(temperature=0, tiktoken_model_name="gpt-3.5-turbo-instruct")

    from transformers import T5Tokenizer, T5ForConditionalGeneration

    # Load the T5 model and tokenizer from Hugging Face
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    def extract_entities_and_relationships(text):
        # Define a prompt for entity and relationship extraction
        prompt = f"Extract the entities and relationships from the following text:\n{text}"

        # Tokenize the prompt and input text
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")

        # Generate output using the model
        output = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)

        # Decode the output to get the text result
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded_output

    prompt = """
You are an expert knowledge graph builder. Parse the following text into a set of nodes and relationships for a knowledge graph. 

For each entity, output its type and name in the format:
EntityType: EntityName

For relationships between entities, output them in the format:
EntityName1 -> EntityName2 [RelationshipType]

If a relationship is not explicitly mentioned, leave it blank or note that there is no explicit relationship.

Input text:
{text}
          """
    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["text"]
    )



    # llm = ChatOllama(
    #     model="llama3.1",
    #     temperature=0
    # )
    #
    # response = ollama.chat(model="llama3.1", prompt_template=prompt_template)

    # Example Wikipedia page title
    page_title = wikipedia_config["page_title"]

    wiki_loader = WikipediaDocumentLoader(page_title=page_title)
    raw_documents = wiki_loader.load()
    documents = wiki_loader.split_document(raw_documents[:3])
    input_text = " ".join([doc.page_content for doc in documents])

    result = extract_entities_and_relationships(input_text)
    rendered_prompt = prompt_template.format(text=input_text)
    knowledge_graph_output = llm.invoke(rendered_prompt)

    # Step 2: Parse content into a knowledge graph
    kg_builder = KnowledgeGraphBuilder(llm=llm)
    parsed_output = kg_builder.convert(documents)

    print("Parsed Output:\n", parsed_output)  # Debugging purpose

    with Neo4jEngine(neo4j_config['uri'],
                     neo4j_config['username'],
                     neo4j_config['password']) as engine:
        kg_builder.process_and_store(parsed_output, engine)


if __name__ == "__main__":
    main()

