from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024)

prompt1  = PromptTemplate(
    template = "Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)
prompt2 = PromptTemplate(
    template = "Generate 5 short question answers from the following text:\n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge provided notes and quize into a single document \n notes: {notes} \n quize: {quize}",
    input_variables=["notes", "quize"]
)

