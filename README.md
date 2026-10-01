# AI Python

A small FastAPI app for resume/job-description ingestion and a retrieval-augmented chat assistant.

## What it does

- Accepts resume text, job description, and self-description
- Splits the text into chunks and creates embeddings
- Stores the chunks in MongoDB
- Retrieves relevant context for a question
- Uses Gemini + LangChain to answer from that context
- Keeps chat history per session

## Tech stack

- Python
- FastAPI
- MongoDB
- LangChain
- Google Generative AI
- sentence-transformers / embeddings

## Setup

1. Create and activate a virtual environment
2. Install dependencies

```bash
pip install -e .
```

3. Set environment variables

```bash
MONGO_URI="your_mongo_connection_string"
GOOGLE_API_KEY="your_google_api_key"
```

4. Run the app

```bash
uvicorn ai_python:app --reload
```

## API

### POST /ingest
Stores resume, job description, and self-description chunks for an interview ID.

### POST /chat
Returns an answer generated from retrieved context for the given interview and session.

## Project notes

This app is built around a simple RAG flow:

```text
text -> chunk -> embed -> store in MongoDB -> retrieve relevant chunks -> LLM answer
```

The main app entry is in `src/ai_python/__init__.py`.
