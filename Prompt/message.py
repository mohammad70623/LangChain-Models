from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq 
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024)

message =[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about langchain"),
]
result = model.invoke(message)
message.append(AIMessage(content=result.content))

print(message)