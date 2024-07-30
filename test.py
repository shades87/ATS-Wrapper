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
    def __init__(self, num_demographic_features, hidden_size, vocab_size):
        super(DemographicBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.demographic_embedding = nn.Linear(num_demographic_features, hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, demographics, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        demographic_embed = self.demographic_embedding(demographics).unsqueeze(1).expand(-1, sequence_output.size(1), -1)
        combined_output = sequence_output + demographic_embed
        logits = self.classifier(combined_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss

        return logits

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
model = DemographicBERT(num_demographic_features, hidden_size, vocab_size).to(device)
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
        input_ids = batch['input_ids'].to(device)
        summary_ids = batch['summary_ids'].to(device)
        demographics = batch['demographics'].to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
        
        print("input ids shape: " + str(input_ids.shape))
        print("summary ids shape: " + str(summary_ids.shape))
        print("demographics shape" + str(demographics.shape))

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, demographics=demographics, labels=summary_ids)
        
        if isinstance(outputs, tuple):
            logits, loss = outputs
        else:
            logits = outputs
            loss = None  # Handle case where no labels are provided
        
        # Check shapes before reshaping
        print(f"Shape of logits before reshaping: {logits.shape}")
        print(f"Shape of summary_ids before reshaping: {summary_ids.shape}")
        
        # Flatten logits and summary_ids correctly
        logits = logits.view(-1, logits.size(-1))  # Flatten to [batch_size * seq_len, vocab_size]
        summary_ids = summary_ids.view(-1)  # Flatten to [batch_size * target_seq_len]
        
        # Print shapes after reshaping
        print(f"Shape of logits after reshaping: {logits.shape}")
        print(f"Shape of summary_ids after reshaping: {summary_ids.shape}")

        # Ensure summary_ids match the shape of logits
        if logits.size(0) != summary_ids.size(0):
            if logits.size(0) > summary_ids.size(0):
                summary_ids = torch.nn.functional.pad(summary_ids, (0, logits.size(0) - summary_ids.size(0)), "constant", tokenizer.pad_token_id)
            else:
                logits = logits[:summary_ids.size(0), :]

        # Compute loss
        loss = criterion(logits, summary_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    # Save the model weights at the end of each epoch
    save_model(model, f'model_epoch_{epoch + 1}.pth')

# Free up unused memory
torch.cuda.empty_cache()

# Load the model weights
load_model(model, 'model_epoch_10.pth')