from fastapi import FastAPI
from demographicsBackUp import *

app = FastAPI()

#python -m uvicorn api:app --reload
@app.get("/")
async def read_root():
    return {"Message": "Congrats! This is your first API!"}
