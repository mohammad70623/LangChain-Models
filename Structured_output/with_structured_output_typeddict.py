from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-safeguard-20b", temperature=0, max_tokens=1024)

#schema 
class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(""" The Hardware is great but the software feels blooted. There are too many free installed apps that i cannot remove. also ui looks outdated compare to other brands. hopping for a software update to fix this.""")

print(result)
print([result['summary'], result['sentiment']])