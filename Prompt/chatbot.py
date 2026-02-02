from langchain_groq import ChatGroq 
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024)

