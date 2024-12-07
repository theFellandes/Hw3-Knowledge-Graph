import os

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.schema import BaseOutputParser
from langchain_openai import OpenAI
from langchain.output_parsers import StructuredOutputParser
from typing import Dict

# Define the Prompt
prompt_template = """
You are an expert at creating knowledge graphs from text. Use the given input to generate a structured knowledge graph.

Input:
{text}

Output format:
[
    {
        "entity": "<entity_name>",
        "type": "<entity_type>",
        "relationships": [
            {
                "relation": "<relation_name>",
                "related_entity": "<related_entity_name>"
            }
        ]
    },
    ...
]

Now create a knowledge graph:
"""

# Create a PromptTemplate
prompt = PromptTemplate.from_template(prompt_template)

# Create the LLM and Runnable
os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI(temperature=0, max_tokens=250)

chain = prompt | llm

# Define the input
input_text = """
Page: Al-Khwarizmi
Summary: Muhammad ibn Musa al-Khwarizmi (Persian: محمد بن موسى خوارزمی; c. 780 – c. 850), or simply al-Khwarizmi, was a polymath who produced vastly influential Arabic-language works in mathematics, astronomy, and geography...
"""

# Run the chain
if __name__ == "__main__":
    chain.invoke(
        {
            "output_language": "English",
            "input": input_text,
        }
    )