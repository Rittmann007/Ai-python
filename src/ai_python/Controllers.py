from ai_python.Models import InterviewRequest,ChatRequest
from ai_python.UtilFuncs import chunk_text
from ai_python.UtilFuncs import get_chunk_embedding
from fastapi import Request,HTTPException
from fastapi.responses import JSONResponse
from bson import ObjectId

# payload is the JSON body the client sends to your endpoint. request is the full FastAPI request object.
# In controller:
# payload: InterviewRequest gives you the data fields like interviewID, userID, resumeText, etc.
# request: Request gives you access to the HTTP request itself, including request.app.state.mongo_client
# use request when you need app state, headers, query params, client info, and similar request-level details
async def ingestController(payload:InterviewRequest,request:Request):
    client = request.app.state.mongo_client
    collection=client["Genai_resumeChecker"]["ragCollection"]

    # chunk the input
    resumeChunk = chunk_text(payload.resumeText, chunk_size=200, chunk_overlap=30)
    jdChunk = chunk_text(payload.jobDescription, chunk_size=200, chunk_overlap=30)
    sdChunk = chunk_text(payload.selfDescription, chunk_size=200, chunk_overlap=30)

    # get the embedding and create the doc to insert
    resumeDocs = [
        {   
            "interviewID" : ObjectId(payload.interviewID),
            "text" : chunk,
            "embedding" : get_chunk_embedding(chunk)
        } for chunk in resumeChunk
    ]
    jdDocs = [
        {   
            "interviewID" : ObjectId(payload.interviewID),
            "text" : chunk,
            "embedding" : get_chunk_embedding(chunk)
        } for chunk in jdChunk
    ]
    sdDocs = [
        {   
            "interviewID" : ObjectId(payload.interviewID),
            "text" : chunk,
            "embedding" : get_chunk_embedding(chunk)
        } for chunk in sdChunk
    ]

    collection.insert_many(resumeDocs)
    collection.insert_many(jdDocs)
    collection.insert_many(sdDocs)

    return JSONResponse(status_code=200, content={"message": "ingested successfully"})

async def chatController(payload:ChatRequest):
    dd