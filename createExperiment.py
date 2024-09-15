#import
import sys
from openai import OpenAI #Chat GPT 3.5


#declare test number
test = "one"

#declare demographics
age = 0
ed = 0
income = 0
metro = 0
nat = 0


def demographics(age=0, ed=0, income=0, region=0, country=0):
  demo = ""
  match age:
    case 1:
      demo = demo + "User is under 15. "

    case 2:
      demo = demo + "User is between 15 and 35. "

    case 3:
      demo = demo + "User is between 35 and 65. "

    case 4:
      demo = demo + "User is over 65. "

  match ed:
    case 1:
      demo = demo + "User has graduated high school. "
          
    case 2:
      demo = demo + "User has a Bachelor's degree. "

    case 3:
      demo = demo + "User has a PHD. "
      
  match country:
    case 1:
      demo = demo + "User is Australian. "

    case 2:
      demo = demo + "User is from New Zealand. "

    case 3:
      demo = demo + "User is from the UK. "

    case 4:
      demo = demo + "User is from the United States of America. "
          
  match region:
    case 1:
      demo = demo + "User lives in a city. "

    case 2:
      demo = demo + "User lives in a rural area. "

  match income:
    case 1:
      demo = demo + "User earns less than $30000 a year."
    
    case 2:
      demo = demo + "User earns between $30000 and $100000 a year."
        
    case 3:  
      demo = demo + "User earns over $100000 a year."
  
  if demo == "":
    demo = "Summarize the provided article"
  else:
    demo = "Summarize the provide article using language best suited to engage the user. " + demo
  
  return demo   

def demographicsBART(age=0, ed=0, income=0, region=0, country=0):
    demo = ""
    match age:
        case 1:
            demo = demo + "age_under_15"

        case 2:
            demo = demo + "age_15_to_35"

        case 3:
            demo = demo + "age_35_to_65"

        case 4:
            demo = demo + "age_over_65"

    match ed:
        case 1:
            demo = demo + "ed_high_school"
            
        case 2:
            demo = demo + "ed_bachelor_degree"

        case 3:
            demo = demo + "ed_phd"
        
    match country:
        case 1:
            demo = demo + "nat_australia"

        case 2:
            demo = demo + "nat_new_zealand"

        case 3:
            demo = demo + "nat_england"

        case 4:
            demo = demo + "nat_united_states"
            
    match region:
        case 1:
            demo = demo + "region_city"

        case 2:
            demo = demo + "region_rural"

    match income:
        case 1:
            demo = demo + "income_under_$30K"
        
        case 2:
            demo = demo + "income_$30K_to_$100k"
            
        case 3:  
            demo = demo + "income_over_$100k"

demoChatGPT = demographics(age, ed, income, metro, nat)

demoBart = demographicsBART(age=0, ed=0, income=0, region=0, country=0)


#load BART model and weights

#GPT cummarise
def summarise(demo, cont):
  client = OpenAI()
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": demo},
      {"role": "user", "content": cont}
    ]
  )

#generate 10 BART summaries and save them

#generate 10 GPT summaries and save them

for x in range()