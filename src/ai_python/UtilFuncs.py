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
        separators=["\n\n", "\n", " ", ""]# split the chunks based on paragraph,line(if chunk>size),word,character
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
            collection (str): A mongodb collection object
            interviewID (str): id of the interview report
    
        Returns:
            List[{"text": val},{},{}...]: List of dicts.
    """
    query_embedding = get_chunk_embedding(query)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",# name of the pre-built Atlas vector search index on this collection
                "queryVector": query_embedding,# the vector to search with
                "path": "embedding",# the field in each document holding the stored chunk embedding
                "numCandidates": 384,# how many candidate documents the approximate-nearest-neighbor algorithm examines before narrowing down
                "limit": 5,# return only the top 5 closest matches
                "filter": {
                    "interviewID": ObjectId(interviewID)# pre-filters the search to only documents belonging to that specific interview
                    # Note: this requires interviewID to be indexed as a filterable field in the vector index definition
                }
            }
        },
        {
            "$project": {# shapes the output: drops _id, keeps only the text field of each matched chunk.
                "_id": 0,
                "text": 1
            }
        }
    ]
    return list(collection.aggregate(pipeline))