from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np


embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Cox’s Bazar is a popular coastal city in southeastern Bangladesh.",
    "It is famous for having the longest natural sea beach in the world.",
    "Tourists visit Cox’s Bazar to enjoy the beach, seafood, and scenic sunsets.",
    "The area is also close to natural attractions like Himchari and Saint Martin’s Island.",
    "Cox’s Bazar plays an important role in Bangladesh’s tourism and local economy."
]

query = "What is Cox’s Bazar known for?" 

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
score = cosine_similarity([query_embedding], doc_embedding)[0]
index, score= sorted(enumerate(score), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("Similarity Score:", score)

