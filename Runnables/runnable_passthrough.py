from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv() 

model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024) 

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

perser = StrOutputParser() 

prompt2 = PromptTemplate(
    template= "Explain the following joke {text}",
    input_variables=["text"]
)

joke_gen_chain = RunnableSequence(prompt1, model, perser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, model, perser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
response = final_chain.invoke({"topic": "AI"})
print(response)
