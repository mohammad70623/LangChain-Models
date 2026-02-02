from langchain_groq import ChatGroq 
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024)


while True:
    user_input = input('You: ')
    if user_input == 'exit':
        break 
    result = model.invoke(user_input)
    print('AI', result.content)