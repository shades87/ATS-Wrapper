import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import torch.optim as optim
from dataset import *


# Example tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the possible options for each demographic field
age_options = ["Under 15", "15-35", "35-65", "65+"]
education_options = ["High School", "Bachelor's Degree", "PHD"]
metro_options = ["Metro", "Regional"]
country_options = ["Australia", "England", "New Zealand", "United States"]
income_options = ["Under 35K", "35K-100K", "100K+"]

# Function to one-hot encode demographic data
def one_hot_encode(demographics):
    age, ed, metro, country, income = demographics
    age_vector = [1 if age == option else 0 for option in age_options]
    ed_vector = [1 if ed == option else 0 for option in education_options]
    metro_vector = [1 if metro == option else 0 for option in metro_options]
    country_vector = [1 if country == option else 0 for option in country_options]
    income_vector = [1 if income == option else 0 for option in income_options]
    return age_vector + ed_vector + metro_vector + country_vector + income_vector

# Custom Dataset class
class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data[idx]["article"]
        summary = self.data[idx]["summary"]
        demographics = (self.data[idx]["age"], self.data[idx]["ed"], self.data[idx]["metro"], self.data[idx]["nat"], self.data[idx]["income"])

        input_ids = tokenizer.encode(article, padding='max_length', truncation=True, max_length=512, return_tensors="pt").squeeze()
        summary_ids = tokenizer.encode(summary, padding='max_length', truncation=True, max_length=128, return_tensors="pt").squeeze()
        demographics_encoded = torch.tensor(one_hot_encode(demographics), dtype=torch.float)

        return {
            'input_ids': input_ids,
            'summary_ids': summary_ids,
            'demographics': demographics_encoded
        }

# Example custom model
class DemographicBERT(nn.Module):
    def __init__(self, demographic_size):
        super(DemographicBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.demographic_embedding = nn.Linear(demographic_size, 768)
        self.decoder = nn.Linear(768, tokenizer.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input_ids, demographic_info, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        demographic_embed = self.demographic_embedding(demographic_info)
        combined = last_hidden_state + demographic_embed.unsqueeze(1)
        decoded = self.decoder(combined)
        return self.softmax(decoded)

# Function to save model weights
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Function to load model weights
def load_model(model, path):
    model.load_state_dict(torch.load(path))

# Define parameters
num_demographic_features = len(age_options) + len(education_options) + len(metro_options) + len(country_options) + len(income_options)
hidden_size = 768
vocab_size = tokenizer.vocab_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = DemographicBERT(demographic_size=16).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Example data for illustration
data = load_data()

dataset = NewsDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in dataloader:
        #Loading batch
        input_ids = batch['input_ids'].to(device)
        summary_ids = batch['summary_ids'].to(device)
        demographics = batch['demographics'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, demographic_info=demographics, attention_mask=input_ids.ne(tokenizer.pad_token_id))
        
        # Flatten outputs and summary_ids
        outputs = outputs.view(-1, tokenizer.vocab_size)
        summary_ids = summary_ids.view(-1)
        
        # Ensure that the reshaped tensors have matching sizes
        num_valid_elements = min(outputs.shape[0], summary_ids.shape[0])
        outputs = outputs[:num_valid_elements]
        summary_ids = summary_ids[:num_valid_elements]
        
        # Compute loss
        loss = criterion(outputs, summary_ids)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Free up unused memory
torch.cuda.empty_cache()

# Load the model weights
load_model(model, 'model_epoch_10.pth')