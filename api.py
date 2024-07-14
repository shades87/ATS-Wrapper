from fastapi import FastAPI
from demographicsBackUp import *
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
    "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#python -m uvicorn api:app --reload
@app.get("/")
async def read_root():
    return {"Message": "Congrats! This is your first API!"}

class summaryClass(BaseModel):
    url: str
    age: int
    nat: int
    income: int
    ed: int
    city: int

#summarise with chat GPT 3.5
@app.post("/summariseGPT")
async def summarise(demographics: summaryClass):
    summary = "the summary"
    demos = [demographics.age, demographics.city, demographics.ed, demographics.income, demographics.nat]
    article = demographics.url
    summary = flow(demos, article)
    print(summary)
    return summary