from langchain_groq import ChatGroq 
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024)

chat_history = []

while True:
    user_input = input('You: ')
    chat_history.append(user_input)
    if user_input == 'exit':
        break 
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print('AI', result.content)

print("Chat ended.")