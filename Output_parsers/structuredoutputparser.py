from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, response_schema

# Load model
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device=-1,
    pipeline_kwargs={
        "temperature": 0.01,
        "max_new_tokens": 512
    }
)

model = ChatHuggingFace(llm=llm)

schema = [
    response_schema(name='fact_1', description='fact 1 about the topic'),
    response_schema(name='fact_2', description='fact 2 about the topic'),
    response_schema(name='fact_3', description='fact 3 about the topic'),
]

parser= StructuredOutputParser.from_response_schemas(schema)