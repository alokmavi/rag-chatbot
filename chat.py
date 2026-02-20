import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def format_retrieved_documents(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def initialize_chat_interface():
    try:
        # Initialize connections
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            embedding=embeddings
        )
        document_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")
        
        # System instructions
        system_prompt = """
        You are a highly precise assistant for university students. 
        Answer the question based strictly on the provided context. If the answer is absent from the context, explicitly state "I don't know." Do not hallucinate.

        Context:
        {context}

        Question: {question}
        """
        prompt_template = ChatPromptTemplate.from_template(system_prompt)
        
        # Build the RAG pipeline
        rag_pipeline = (
            {"context": document_retriever | format_retrieved_documents, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        print("--- RAG CHATBOT ONLINE (Type 'exit' to quit) ---")
        
        # Interactive execution loop
        while True:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down...")
                sys.exit(0)
                
            print("Querying vector database...")
            llm_response = rag_pipeline.invoke(user_input)
            print(f"Bot: {llm_response}")

    except Exception as runtime_error:
        print(f"System Failure: {runtime_error}")
        sys.exit(1)

if __name__ == "__main__":
    initialize_chat_interface()
