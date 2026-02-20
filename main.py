import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

ml_pipeline = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_db = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            embedding=embedder
        )
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        llm_engine = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")
        
        system_prompt = """
        You are a highly precise academic assistant.
        Answer the question based strictly on the provided context. If the answer is absent, state "I don't know."
        
        Context:
        {context}
        
        Question: {question}
        """
        prompt_tpl = ChatPromptTemplate.from_template(system_prompt)
        
        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])
            
        ml_pipeline["rag"] = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_tpl
            | llm_engine
            | StrOutputParser()
        )
        print("--- PIPELINE INITIALIZED ---")
        yield
    except Exception as initializationError:
        print(f"System Failure: {initializationError}")
        raise
    finally:
        ml_pipeline.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryPayload(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)

@app.post("/api/v1/ask")
async def process_query(payload: QueryPayload):
    rag_chain = ml_pipeline.get("rag")
    if not rag_chain:
        raise HTTPException(status_code=503, detail="Model pipeline is not ready.")
    
    try:
        answer = rag_chain.invoke(payload.question)
        return {"status": "success", "answer": answer}
    except Exception as inferenceError:
        raise HTTPException(status_code=500, detail=str(inferenceError))
