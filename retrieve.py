import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# 1. Connect to the Index
print("--- CONNECTING TO DATABASE ---")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

# 2. Ask a Question
query = "What represents the user in this document?"  # <--- WE WILL CHANGE THIS LATER
print(f"\nQuery: {query}\n")

# 3. Search (The Magic)
print("--- SEARCHING ---")
results = vectorstore.similarity_search(query, k=2) # Get top 2 matches

# 4. Show Results
for i, doc in enumerate(results):
    print(f"\n[MATCH {i+1}]")
    print(doc.page_content)
    print("-" * 50)
