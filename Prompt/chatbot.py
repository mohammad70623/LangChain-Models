from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024)

chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break 
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI', result.content)

print("Chat ended.")