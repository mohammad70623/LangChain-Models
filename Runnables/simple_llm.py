from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024)

prompt = PromptTemplate(
    template='Suggest a blog title about {topic}',
    input_variables=['topic']
) 

topic = input('input a topic: ')

formatted_prompt = prompt.format(topic=topic)

blog_title = llm.invoke(formatted_prompt)

print(blog_title)

