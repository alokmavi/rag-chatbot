# Context-Aware Syllabus Chatbot (RAG)

A full-stack Retrieval-Augmented Generation (RAG) microservice designed to ingest, embed, and query complex academic syllabus data. The system utilizes a vector-search architecture to provide precise, hallucination-free answers to student queries, referencing specific course codes and credit distributions.

## Architecture

The application is built as a containerized microservice:

* **Ingestion Pipeline:** Python scripts parse raw PDF data, chunk text based on semantic boundaries, and generate embeddings using **Google Gemini 1.5**.
* **Vector Database:** Embeddings are stored in **Pinecone** for high-performance similarity search (k-NN).
* **Backend API:** An asynchronous **FastAPI** server handles query processing, retrieving relevant context chunks from Pinecone before injecting them into the LLM prompt.
* **Frontend:** A lightweight, semantic HTML/JS interface that consumes the API via REST, rendering Markdown responses in real-time.

## Tech Stack

* **Core:** Python 3.11, JavaScript (Vanilla)
* **Framework:** FastAPI (Uvicorn server)
* **AI/ML:** LangChain, Google Gemini API (Embeddings + Chat)
* **Database:** Pinecone Vector DB
* **Infrastructure:** Docker, Docker Compose

## Setup & Installation

### Prerequisites
* Docker & Docker Desktop
* API Keys for Google AI Studio and Pinecone

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/alokmavi/rag-chatbot.git](https://github.com/alokmavi/rag-chatbot.git)
    cd rag-chatbot
    ```

2.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_google_key
    PINECONE_API_KEY=your_pinecone_key
    PINECONE_INDEX_NAME=your_index_name
    ```

3.  **Run with Docker**
    Build and start the container:
    ```bash
    docker build -t rag-api .
    docker run -p 8000:8000 --env-file .env rag-api
    ```

4.  **Access the UI**
    Open `index.html` in any modern web browser.

## API Usage

**Endpoint:** `POST /api/v1/ask`

**Payload:**
```json
{
  "question": "What are the credit requirements for the Data Structures course?"
}
```

**Response:**
```json
{
  "status": "success",
  "answer": "The Data Structures course (BTCS208T) carries 3 credits..."
} ```
