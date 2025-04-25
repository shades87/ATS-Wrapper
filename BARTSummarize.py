import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration
from GPTSummarise import demographics
from huggingface_hub import login
from dotenv import load_dotenv

#Basically a model to load just for the API
def summariseBART(demoArr, article):
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    repo_name = "shadyshadyshades/ATS-Wrapper"
    demos = demographics(demoArr[0], demoArr[1],demoArr[2],demoArr[3], demoArr[4])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BartForConditionalGeneration.from_pretrained('./weights/BARTLarge1/')
    model.load_state_dict(
        BartForConditionalGeneration.from_pretrained(
            repo_name,
            use_auth_token=token,
            return_dic=True
        ).state_dict()
    )
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    combined_input = demos + " " + article
    print(combined_input)
    inputs = tokenizer(combined_input, max_length = 1024, return_tensors="pt", truncation=True).to(device)

    model.to(device)    

    with torch.no_grad():
        summary_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=1024, num_beams=4, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
