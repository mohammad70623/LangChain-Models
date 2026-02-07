from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv() 

model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024) 

prompt1 = PromptTemplate(
    template="Write a tweet about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template= "Write a LinkedIn post about {topic}",
    input_variables=["topic"]
)

perser = StrOutputParser()


parallel_chain = RunnableParallel(
    {
    "tweet": RunnableSequence(prompt1, model, perser),
    "linkedin": RunnableSequence(prompt2, model, perser)
    }
)
response = parallel_chain.invoke({"topic": "AI"})
print(response)