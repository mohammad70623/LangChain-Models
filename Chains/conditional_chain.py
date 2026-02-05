from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024)

prompt1 = PromptTemplate(
    template= "Classify the sentimant of the following feedback text into positive and negetive \n {feedback}",
    input_variables=["feedback"]
)

parser = StrOutputParser()

classifier_chain = prompt1 | model | parser 
print(classifier_chain.invoke({"feedback": "The product quality is excellent and delivery was prompt."}))