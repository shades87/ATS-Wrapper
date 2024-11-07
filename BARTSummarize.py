import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from GPTSummarise import demographics

#Basically a model to load just for the API
def summariseBART(demoArr, article):

    demos = demographics(demoArr[0], demoArr[1],demoArr[2],demoArr[3], demoArr[4])
    device = "cuda"
    model = BartForConditionalGeneration.from_pretrained('weights/BARTLarge1/')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    combined_input = demos + " " + article
    print(combined_input)
    inputs = tokenizer(combined_input, max_length = 1024, return_tensors="pt", truncation=True).to(device)

    model.to(device)    

    with torch.no_grad():
        summary_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=1024, num_beams=4, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
