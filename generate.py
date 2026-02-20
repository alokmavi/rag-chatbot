import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Setup Vector DB (The Memory)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Setup LLM (The Brain)
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

# 3. The Prompt (The Instructions)
template = """
You are a helpful assistant for university students. 
Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. The Chain (The Assembly Line)
# Retriever -> Format Docs -> Prompt -> LLM -> String Output
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Run It
question = "What are the learning outcomes for Unit 1?"
print(f"Question: {question}\n")
print("Thinking...")

response = rag_chain.invoke(question)
print(f"\nAnswer:\n{response}")
