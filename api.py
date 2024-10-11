from fastapi import FastAPI
from GPTSummarise import *
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from BERTSummarise import *
from BARTSummarize import *
import google.generativeai as genai
from dotenv import load_dotenv


app = FastAPI()

#We're hosting a svelte app on a different address so Cross Origin api access is needed
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

#my first time using fastapi so let's save the command
#python -m uvicorn api:app --reload
@app.get("/")
async def read_root():
    return {"Message": "Congrats! This is your first API!"}

#classes to simplify passing Json in body to api
# I wish I did Extended Distributed Computing before I started my dissertation 
class urlClass(BaseModel):
    url: str

class summaryClass(BaseModel):
    url: str
    age: int
    nat: int
    income: int
    ed: int
    city: int

class summaryForBART(BaseModel):
    url: str
    age: str
    nat: str
    income: str
    ed: str
    city: str

#summarise with chat GPT 3.5
#If you don't have a API key saved in your environment the whole thing won't work
@app.post("/summariseGPT")
async def summariseA(demographics: summaryClass):
    summary = "the summary"
    demos = [demographics.age, demographics.city, demographics.ed, demographics.income, demographics.nat]
    article = demographics.url
    summary = flow(demos, article)
    print(summary)
    return {"message": summary}

#summarise with custom model built on top of BERT
#I left this in on purpose even though it doesn't work
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
    try:
        article = get_article_text(demographics.url)
        #debug check article
        #print("Article: " + article)
        #debug, check one_hot list
        #print(one_hot)


        summary = bertSummarize(article, one_hot)
        #print(summary)
        #Does fastapi return a code or should I return OK? I need to check this
        return {"message": summary}
    except:
        return {"message": "Error downloading article: Check the URL" }

#Summarise for BART
#BART is the slowest summarizer, I should consider loading the model once when the api is first loaded and passing the model in
#to save time
#The first time BART is called it normally times out, some change is needed
@app.post("/summariseBART")
async def summariseC(demographics: summaryClass):
    try:
            
        article = get_article_text(demographics.url)
        user = ""
        age = ""
        income = ""
        nat = ""
        ed = ""
        metro = ""

        #Tried more than one way to input demographics

        #if demographics.age:
            #age = "age_" +  demographics.age
            #age.replace(" ", "_")
            #user = user + age + " "

        #if demographics.income:
            #income = "income_" + demographics.income
            #income.replace(" ", "_")
            #user = user + income + " "

        #if demographics.nat: 
            #nat = "nat_" + demographics.nat
            #nat.replace(" ", ",")
            #user = user + nat + " "

        #if demographics.ed:
            #ed = "ed_" + demographics.ed
            #ed.replace(" ", ",")
            #user = user + ed + " "

        #if demographics.city:
            #metro = "metro_" + demographics.city
            #metro.replace(" ", ",")
            #user = user + metro + " "

        #print(user)

        demos = [demographics.age, demographics.city, demographics.ed, demographics.income, demographics.nat]


        summary = summariseBART(demos, article)

        return {"message": summary}
    except:
        return {"message": "Error downloading article: Check the URL"}

#Gemini AI
#If you don't have a Google AI Key the whole page won't work, comment it out if this is the case
@app.post("/summarizeGemini")
async def summariseD(demographic: summaryClass):
    load_dotenv()
    try:
        article = get_article_text(demographic.url)
        demoArr = [demographic.age, demographic.city, demographic.ed, demographic.income, demographic.nat]
        demos = demographics(demoArr[0], demoArr[1],demoArr[2],demoArr[3], demoArr[4])

        THE_GOOGLE_API_KEY = os.getenv('GEMINI_KEY')
        print(repr(THE_GOOGLE_API_KEY))
        genai.configure(api_key=THE_GOOGLE_API_KEY)

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(demos+article)
        return { "message": response.text}
    except:
        return{"message": "Unable to download article: Check the URL is correct"}
