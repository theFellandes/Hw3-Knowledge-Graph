import csv
import io
import json
from dataclasses import dataclass, field
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from typing_extensions import deprecated

from internal.llm.llm import LLMBase


@dataclass
class ChatGptLLM(LLMBase):
    chunks: list
    model: str = field(default="gpt-4o")
    temperature: int = field(default=0)
    max_tokens: int = field(default=500)

    @staticmethod
    def __post_init__():
        LLMBase.enable_cache()

    @staticmethod
    def csv_cleaner(response_content):
        cleaned_data = []

        for item in response_content:
            try:
                source = item.get("```csv")
                relationship, target = item.get(None, [None, None])
                if source and relationship and target:
                    cleaned_data.append([source, relationship, target])
            except Exception as e:
                continue

        # Write to a CSV file
        with open("relationships.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["source", "relationship", "target"])  # Header
            writer.writerows(cleaned_data)

        return cleaned_data

    def generate_relationships_csv(self):
        llm = ChatOpenAI(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)

        relationships = []
        for chunk in self.chunks:
            # Define the prompt for CSV output
            prompt = f"""
            Analyze the following text and extract entities and relationships in CSV format.
            The CSV should have the following columns: 'source', 'relationship', 'target'.

            Text: {chunk.page_content}

            Example output:
            source,relationship,target
            Marcus Aurelius,wrote,Meditations
            Marcus Aurelius,was born in,Rome

            Your output should be a CSV with the columns 'source', 'relationship', and 'target'.
            """
            try:
                # Call the LLM
                response = llm([{"role": "user", "content": prompt}])
                response_content = response.content.strip()

                # Parse the CSV response
                csv_reader = csv.DictReader(io.StringIO(response_content))
                for row in csv_reader:
                    relationships.append(row)
            except Exception as e:
                print(f"Error parsing response: {e}")
                print(f"Response received: {response.content.strip()}")
        return self.csv_cleaner(relationships)

    @deprecated("Old version of the function, use generate_relationships_csv instead.")
    def generate_relationships(self):
        llm = ChatOpenAI(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)

        relationships = []
        for chunk in self.chunks:
            # Define the prompt for structured JSON output
            prompt = f"""
            Analyze the following text and extract entities and relationships in JSON format.
            Please return only valid JSON data, and make sure the keys are 'source', 'relationship', and 'target'.
            Each relationship should be a dictionary with these three keys.

            Text: {chunk.page_content}

            Example output:
            [
                {{
                    "source": "Marcus Aurelius",
                    "relationship": "wrote",
                    "target": "Meditations"
                }},
                {{
                    "source": "Marcus Aurelius",
                    "relationship": "was born in",
                    "target": "Rome"
                }}
            ]

            Your output should be a list of relationships in JSON format, and nothing else.
            """
            try:
                # Get the response from the model
                response = llm([{"role": "user", "content": prompt}])
                response_content = response.content.strip()

                # Remove any unwanted characters like backticks or extra text
                cleaned_response = response_content.replace('```csv', '').replace('```', '').strip()

                # Debugging: Print the cleaned response
                print(f"Cleaned response: {cleaned_response}")

                # Parse the cleaned response into JSON
                relationships_chunk = json.loads(cleaned_response)

                # Add the relationships from this chunk
                relationships.extend(relationships_chunk)

            except Exception as e:
                print(f"Error parsing response: {e}")
                print(f"Response received: {response.content.strip()}")

        # Debugging: Print all relationships before storing
        print(f"Generated relationships: {relationships}")
        return relationships

    def generate_summary(self):
        llm = ChatOpenAI(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)
        llm_transformer = LLMGraphTransformer(llm=llm)

        graph_documents = llm_transformer.convert_to_graph_documents(self.chunks)

        return graph_documents

