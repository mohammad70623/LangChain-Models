from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv() 

def word_count(text):
    return len(text.split())

model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024) 

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

perser = StrOutputParser() 

joke_gen_chain = RunnableSequence(prompt, model, perser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
response = final_chain.invoke({"topic": "AI"})
print(response)