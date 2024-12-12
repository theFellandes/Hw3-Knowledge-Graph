from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Tuple, List


# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

    @staticmethod
    def get_entity_chain(llm):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting organization and person entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )

        return prompt | llm.with_structured_output(Entities)
