import sys
from openai import OpenAI #Chat GPT 3.5

from newspaper import Article 
from newspaper import Config

#Functions to create and load the dataset
#I intend to use synthetic data to train the BERT model
from demographicsBackUp import demographics

#Load the data in the .txt files in /dataset 
def data():
    
    data = [
    {"article": "Some news article text", "summary": "summary", "age": "Under 15", "gender": "male", "education": "college", "region": "north", "interests": "technology"},
    {"article": "Another article", "summary": "Another summary", "age_group": "25-34", "gender": "female", "education": "high_school", "region": "south", "interests": "sports"},
    # I need more data entries...
    ]

    return data

#Programatically create data to summarise based on all possible demographic information
#Create the summaries using my chatgpt model
def create_data(URL):

    #Grab the URL, create a summary for every possible demographic info
    #Every demographic info
    #Ed: 3 options, Age: 4 options, Nat: 4 options, Metro: 2 options, Income: 3 Options

    articleURL = URL
    ed = 0;
    age = 0;
    nat = 0;
    metro = 0;
    income = 0;

    #1200 permutations
    for a in range(4):
        ed = a
        print("a: " + str(a))
        for b in range(5):
            age = b
            for c in range(5):
                nat = c
                for d in range(3):
                    metro = d
                    for e in range(4):
                        income = 4
                        user = demographics(age, ed, income, metro, nat)

                      
create_data("Fake URL")