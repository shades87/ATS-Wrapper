from fastapi import FastAPI
from demographicsBackUp import *
from pydantic import BaseModel

app = FastAPI()

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

@app.post("/summarise")
async def summarise(summary: summaryClass):
    
    return summary