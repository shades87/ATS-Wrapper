from fastapi import FastAPI
from GPTSummarise import *
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from BERTSummarise import *

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
async def summariseA(demographics: summaryClass):
    summary = "the summary"
    demos = [demographics.age, demographics.city, demographics.ed, demographics.income, demographics.nat]
    article = demographics.url
    summary = flow(demos, article)
    print(summary)
    return {"message": summary}

#summarise with custom model built on top of BERT
@app.post("/summariseBERT")
async def summariseB(demographics: summaryClass):
    #order of age, ed, nat, income, city
    #start of index age:0, ed:4, nat: 7, income: 11, metro: 13
    one_hot = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    if demographics.age > 0:
        one_hot[demographics.age] = 1
    
    if demographics.ed > 0:
        one_hot[3+ demographics.ed] = 1

    if demographics.nat > 0:
        one_hot[6 + demographics.nat] = 1

    if demographics.income > 0:
        one_hot[10+ demographics.income] = 1 

    if demographics.city > 0:
        one_hot[12+ demographics.city] = 1

    print("URL: " + demographics.url)
    article = get_article_text(demographics.url)
    #debug check article
    print("Article: " + article)
    #debug, check one_hot list
    print(one_hot)


    summary = bertSummarize(article, one_hot)
    return {"message": summary}
    