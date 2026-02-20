import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("--- 1. LOADING PDF ---")
loader = PyPDFLoader("data.pdf")
raw_docs = loader.load()
print(f"Loaded {len(raw_docs)} pages.")

print("--- 2. SPLITTING TEXT ---")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_docs)
print(f"Split into {len(documents)} chunks.")

print("--- 3. EMBEDDING & UPLOADING ---")
# WE FOUND THE CORRECT MODEL NAME:
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

PineconeVectorStore.from_documents(
    documents,
    embeddings,
    index_name=os.getenv("PINECONE_INDEX_NAME")
)
print("--- SUCCESS: PDF INGESTED! ---")
