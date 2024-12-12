import os

from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from typing import Tuple, List

from internal.langchain.Queries import Queries
from internal.llm.Entities import Entities

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "your_password"
graph = Neo4jGraph()


def is_database_empty() -> bool:
    """
    Checks whether the Neo4j database is empty.
    Returns True if the database is empty, False otherwise.
    """
    query = """
    MATCH (n)
    RETURN COUNT(n) AS node_count
    """
    result = graph.query(query)
    node_count = result[0]["node_count"] if result else 0
    return node_count == 0


def wikipedia_loader(llm_transformer: LLMGraphTransformer, flag: bool = True):
    if not flag:
        return
    # Read the wikipedia article
    raw_documents = WikipediaLoader(query="Marcus Aurelius").load()
    # Define chunking strategy
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents[:3])
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


def generating_chain():
    _template = """Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    condense_question_prompt = PromptTemplate.from_template(_template)

    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | condense_question_prompt
            | ChatOpenAI(temperature=0)
            | StrOutputParser(),
        ), RunnableLambda(lambda x: x["question"]),
    )

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            RunnableParallel(
                {
                    "context": _search_query | query_generator.retriever,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain


def chat_with_bot(chain):
    print("Chatbot: Hello! Ask me anything. Type 'exit' to quit.")
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        response = chain.invoke(
            {"question": user_input, "chat_history": chat_history}
        )
        print(f"Chatbot: {response}")

        # Update chat history
        chat_history.append((user_input, response))


if __name__ == '__main__':
    load_dotenv()

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    llm_transformer = LLMGraphTransformer(llm=llm)
    wikipedia_loader(llm_transformer, is_database_empty())

    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

    # Retriever
    graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    entity_chain = Entities.get_entity_chain(llm)
    query_generator = Queries(entity_chain, vector_index, graph)

    print(query_generator.structured_retriever("Who is Marcus Aurelius?"))
    chat_with_bot(generating_chain())
