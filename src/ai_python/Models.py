from pydantic import BaseModel

class InterviewRequest(BaseModel):
    interviewID: str
    userID: str
    resumeText: str
    jobDescription: str
    selfDescription: str

class ChatRequest(BaseModel):
    interviewID: str
    userID: str
    message: str