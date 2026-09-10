import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from bson import ObjectId

hf_token = os.environ.get("HF_TOKEN")

# Initialize the embedding model (you can choose any supported model)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": hf_token}
)

def chunk_text(input_text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Splits a string into chunks using LangChain's RecursiveCharacterTextSplitter.

    Args:
        input_text (str): The text to split.
        chunk_size (int): Maximum size of each chunk (default 500).
        chunk_overlap (int): Overlap between chunks (default 50).

    Returns:
        List[str]: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(input_text)
    return chunks


def get_chunk_embedding(chunk: str):
    """
    Takes a single text chunk and returns its embedding vector.
    
    Args:
        chunk (str): The text chunk to embed.
    
    Returns:
        List[float]: Embedding vector for the chunk.
    """
    embedding = embedding_model.embed_query(chunk)
    return embedding

# Example usage
# vector = get_chunk_embedding(result[0])
# print("Embedding length:", len(vector))
# print("First 10 values:", vector[:10])

def get_query_results(query: str,collection,interviewID):
    """
        give relevent chunks related to query filtered by interviewID
    
        Args:
            query (str): The query of the user.
            collection (str): the collection var of the db collection querying on
            interviewID (str): id of the interview report
    
        Returns:
            List[{"text": val}]: List of dicts.
    """
    query_embedding = get_chunk_embedding(query)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 384,
                "limit": 5,
                "filter": {
                    "interviewID": ObjectId(interviewID)
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1
            }
        }
    ]
    return list(collection.aggregate(pipeline))