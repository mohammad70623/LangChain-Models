from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024)

prompt1  = PromptTemplate(
    template = "Generate a details report on {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template = "Generate 5 pointer summary from the following text:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"topic": "Unemployment in Bangladesh"})
print(result)