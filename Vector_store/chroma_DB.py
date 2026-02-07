from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document



# Create LangChain documents for Bangladesh Cricket players

doc1 = Document(
    page_content="Shakib Al Hasan is one of the greatest all-rounders in world cricket. Known for his consistency with both bat and ball, he has been a key player for Bangladesh in all formats.",
    metadata={"team": "Dhaka Capitals"}
)

doc2 = Document(
    page_content="Tamim Iqbal is Bangladesh's most reliable opening batsman. Famous for his aggressive starts and leadership, he has played a crucial role in Bangladesh’s rise in international cricket.",
    metadata={"team": "Chattogram Challengers"}
)

doc3 = Document(
    page_content="Mushfiqur Rahim is one of the most experienced cricketers in Bangladesh. Known for his fighting spirit, wicketkeeping skills, and match-winning innings under pressure.",
    metadata={"team": "Comilla Victorians"}
)

doc4 = Document(
    page_content="Mustafizur Rahman, popularly known as The Fizz, is renowned for his deadly cutters and variations. He is one of Bangladesh’s most successful fast bowlers in limited-overs cricket.",
    metadata={"team": "Fortune Barishal"}
)

doc5 = Document(
    page_content="Mahmudullah Riyad is a dependable middle-order batsman and handy off-spin bowler. His calm temperament and finishing ability have won Bangladesh many crucial matches.",
    metadata={"team": "Khulna Tigers"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='my_chroma_db',
    collection_name='sample'
)

# add documents
vector_store.add_documents(docs)

# view documents
vector_store.get(include=['embeddings','documents', 'metadatas'])

# search documents
vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)

# search with similarity score
vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)

# meta-data filtering
vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Dhaka Capitals"}
)

