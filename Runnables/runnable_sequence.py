from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv() 

model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024)

prompt = PromptTemplate(
    template= "Write a jokes about {topic}",
    input_variables=["topic"],
)
perser = StrOutputParser()

chain = RunnableSequence( prompt, model, perser)
response = chain.invoke({"topic": "AI"})
print(response)