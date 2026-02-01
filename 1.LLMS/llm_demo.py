from langchain_groq import ChatGroq 
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0)
response = llm.invoke("what is the capital of bangladesh?")
print(response)