#import
import os
from openai import OpenAI #Chat GPT 3.5
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
import torch

#declare test number
test = "six"

#declare demographics
##Record the participants and the order they participated
##Laluune one ## Jess two ## Sheddy three ## Jayne 4 ## Deb 5
##Leo 6 ##Rory 7 ## Jones 8
age = 2
ed = 1
income = 1
metro = 1
nat = 1




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
    return demo

demoChatGPT = demographics(age, ed, income, metro, nat)

demoBart = demographicsBART(age, ed, income, metro, nat)


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
    a = completion.choices[0].message.content
    return a

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
device = 'cuda'

def summarizeBART(article, demographics):
    weights = os.path.abspath("weights/BART")
    model = BartForConditionalGeneration.from_pretrained('weights/BART', local_files_only=True)
    model.to(device)
    combined_input = demographics + " " + article
    inputs = tokenizer(combined_input, max_length = 1024, return_tensors="pt", truncation=True).to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=150, num_beams=4, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

#generate 10 GPT summaries and save them
f = open("Experiment/one.txt", 'r')
p = open("Experiment/"+test+"/chatgpt.txt", "w")
for x in range(10):
    string = f.readline()
    print(string)
    summary = summarise(string, demoChatGPT)
    p.write(summary + '\n')
    print(summary)

f.close()
p.close()
#generate 10 BART summaries and save them
#open file
f = open("Experiment/two.txt", "r")
p = open("Experiment/"+test+"/two.txt", "w")
for x in range(10):
   string = f.readline()
   summary = summarizeBART(string, demoBart)
   p.write(summary + "\n")
   print(summary)


f.close()
p.close()