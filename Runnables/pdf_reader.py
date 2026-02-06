# pdf_reader.py
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


pdf_path = "/home/rafat/Mohammad/LangChain_Models/Runnables/MOHAMMAD_CV.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()

query = "What are the key takeaways from the documents?"
retrieved_docs = retriever.invoke(query)

retrieved_text = "\n".join(doc.page_content for doc in retrieved_docs)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=1024,
    api_key=GROQ_API_KEY
)

prompt = f"""
Based on the following retrieved text, answer the question.

Question:{query}
Context:{retrieved_text}
"""

answer = llm.invoke(prompt)

print(answer.content)
