import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import torch.optim as optim
from dataset import *

#I asked ChatGPT for help using One Hot Encoding to train BERT and used the results to help build this 

# Load the dataset that I've created
data = data()

df = pd.DataFrame(data)

# One-hot encode demographic information
demographic_fields = ['age', 'ed', 'nat', 'income', 'metro']
one_hot_encoded_demographics = pd.get_dummies(df[demographic_fields])

# Combine the original dataframe with the one-hot encoded demographic data
df = pd.concat([df, one_hot_encoded_demographics], axis=1)
df = df.drop(columns=demographic_fields)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class DemographicBERT(nn.Module):
    def __init__(self, demographic_size):
        super(DemographicBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.demographic_embedding = nn.Linear(demographic_size, 768)
        self.decoder = nn.Linear(768, tokenizer.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input_ids, demographic_info):
        outputs = self.bert(input_ids)
        last_hidden_state = outputs.last_hidden_state
        demographic_embed = self.demographic_embedding(demographic_info)
        combined = last_hidden_state + demographic_embed.unsqueeze(1)
        decoded = self.decoder(combined)
        return self.softmax(decoded)

class NewsDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        article_tokens = self.tokenizer.encode(row['article'], truncation=True, padding='max_length', max_length=512)
        summary_tokens = self.tokenizer.encode(row['summary'], truncation=True, padding='max_length', max_length=128)
        demographic_info = row[5:].values.astype(float)  # Adjust the slicing as per your dataframe structure
        
        return {
            'input_ids': torch.tensor(article_tokens, dtype=torch.long),
            'summary_ids': torch.tensor(summary_tokens, dtype=torch.long),
            'demographics': torch.tensor(demographic_info, dtype=torch.float)
        }

# Prepare dataset and dataloader
dataset = NewsDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model, loss, and optimizer
model = DemographicBERT(demographic_size=len(one_hot_encoded_demographics.columns))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids']
        summary_ids = batch['summary_ids']
        demographics = batch['demographics']
        
        optimizer.zero_grad()
        outputs = model(input_ids, demographics)
        loss = criterion(outputs.view(-1, tokenizer.vocab_size), summary_ids.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Save the model
model_path = "demographic_bert_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Load the model
model = DemographicBERT(demographic_size=len(one_hot_encoded_demographics.columns))
model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded successfully")

# Inference function
def summarize(article, demographic_info, model, tokenizer):
    input_ids = tokenizer.encode(article, return_tensors='pt')
    demographic_tensor = torch.tensor(demographic_info, dtype=torch.float).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids, demographic_tensor)
    summary_ids = torch.argmax(outputs, dim=-1)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage
article = "A new breakthrough in technology..."
demographic_info = [1, 0, 0, 1, 0]  # Example one-hot encoded demographics
summary = summarize(article, demographic_info, model, tokenizer)
print(summary)