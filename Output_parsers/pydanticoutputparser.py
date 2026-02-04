from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device=-1,  # CPU
    pipeline_kwargs={
        "temperature": 0.01,
        "max_new_tokens": 512
    }
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(gt=18, description="The age of the person")
    city: str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template="""
Give the name, age, and city of a {place} from Asia
Respond in strict JSON format with actual values:
{{
  "name": "...",
  "age": ...,
  "city": "..."
}}
""",
    input_variables=["place"],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)


chain = template | model | parser

result = chain.invoke({'place': 'Bangladesh'})
print(result.name, result.age, result.city)
