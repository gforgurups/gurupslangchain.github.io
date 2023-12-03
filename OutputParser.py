from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


class Person(BaseModel):
    summary: str = Field(description="Summary of the person")
    facts: List[str] = Field(description="Interesting facts about the person")
    topics_of_interests: List[str] = Field(
        description="Topics that may interests the person"
    )
    ice_breakers: List[str] = Field(
        description="Create Ice breakers to open a conversation with the person"
    )

    def to_dict(self):
        return {
            "summary": self.summary,
            "facts": self.facts,
            "topics_of_interest": self.topics_of_interests,
            "ice_breakers": self.ice_breakers,
        }


person_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=Person)
