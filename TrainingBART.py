import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from tqdm import tqdm
from dataset import *

# Custom Dataset for fine-tuning
class SummarizationDataset(Dataset):
    def __init__(self, articles, summaries, demographics, tokenizer, max_length=1024):
        self.articles = articles
        self.summaries = summaries
        self.demographics = demographics
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        combined_input = self.demographics[idx] + " " + self.articles[idx]
        inputs = self.tokenizer(combined_input, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        
        #train on the full summary with padding for shorter ones
        targets = self.tokenizer(self.summaries[idx], max_length=256, padding="max_length", truncation=True, return_tensors="pt")


        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze()
        }

# Load BART tokenizer and model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Articles
data = loadForBARTTwo();
articles = data.get("articles")
summaries = data.get("summaries")
demographics = data.get("demographics")

# Prepare dataset and dataloader
dataset = SummarizationDataset(articles, summaries, demographics, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Fine-tuning settings
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

# Fine-tuning loop
model = BartForConditionalGeneration.from_pretrained('weights\BARTLarge0')
model.to(device)
model.train()
def trainBART():
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader) + 3
        print(f"Epoch {epoch+1} finished with loss: {avg_epoch_loss:.4f}")
        save_number = epoch + 1
        model_path = "weights/BARTLarge"+str(save_number)
        model.save_pretrained(model_path)

# Save fine-tuned model
trainBART()

#print(f"Model saved to {model_save_path}")

# Example usage after fine-tuning
model.eval()

def summarizeBART(article, demographics):
    model = BartForConditionalGeneration.from_pretrained('weights/BARTLarge2/')
    combined_input = demographics + " " + article
    inputs = tokenizer(combined_input, max_length = 1024, return_tensors="pt", truncation=True).to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=256, num_beams=4, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary