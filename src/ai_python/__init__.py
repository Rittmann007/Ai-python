from contextlib import asynccontextmanager
from fastapi import FastAPI
from ai_python.Controllers import ingestController,chatController
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.environ.get("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi("1"))

@asynccontextmanager # this section will run before the app actually starts
async def lifespan(app: FastAPI):
    app.state.mongo_client = client # store the Mongo client on the app during lifespan, then read it inside the controller.
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!", flush=True)
    except Exception as e:
        print(e, flush=True)
    yield

app = FastAPI(lifespan=lifespan)

#routes

# root
@app.get("/")
async def root():
    return {"message": "Hello World"}

# ingest
app.post("/ingest")(ingestController)

# chat
app.post("/chat")(chatController)

