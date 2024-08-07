import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import torch.optim as optim
from dataset import *

print(torch.__version__)
print("")
print("")
print(torch.cuda.is_available())  # Should print True if GPU is available
print(torch.cuda.current_device())  # Should print the current GPU device ID
print(torch.cuda.get_device_name(0))  # Should print the name of the GPU
print("")
print("")
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

# Initialize the model
model = DemographicBERT(demographic_size=16).to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Example data for illustration
data = load_data()

dataset = NewsDataset(data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True) #original batch size is 2 but lets try 16

# Training loop
for epoch in range(10):
    i=1
    print("Epoch: " + str(epoch))
    model.train()
    total_loss = 0
    for batch in dataloader:

        #Loading batch

        input_ids = batch['input_ids'].to('cuda')
        summary_ids = batch['summary_ids'].to('cuda')
        demographics = batch['demographics'].to('cuda')
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, demographic_info=demographics, attention_mask=input_ids.ne(tokenizer.pad_token_id))
        

        print("Batch input_ids shape:", input_ids.shape)  # Should be (16, 512)
        print("Batch summary_ids shape:", summary_ids.shape)  # Should be (16, 128)
        print("Batch demographics shape:", demographics.shape)  # Should be (16, 16)

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
        if i > 16:
            break
        i+=1

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Free up unused memory
torch.cuda.empty_cache()

# Load the model weights
#load_model(model, 'model_epoch_10.pth')

def summarize(article, demographic_info, model, tokenizer):
    input_ids = tokenizer.encode(article, return_tensors='pt').to("cuda")
    attention_mask = torch.tensor([1] * input_ids.size(-1) + [0] * (512 - input_ids.size(-1)), dtype=torch.long).unsqueeze(0).to("cuda")  # attention mask
    demographic_tensor = torch.tensor(demographic_info, dtype=torch.float).unsqueeze(0).to("cuda")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, demographics=demographic_tensor)
    summary_ids = torch.argmax(outputs, dim=-1)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage
article = "SW will ban ‘no fault’ evictions, bringing the state in line with other parts of the country, including Victoria. Landlords currently have the power to end a lease without giving their tenant a reason. However, the NSW Government will introduce legislation to ban this practice. Notice periods for evictions will also be extended under the plan. The State Government said the reforms are set to be rolled out “early next year”. Renting in NSW There are two types of tenancy agreements (leases) for NSW rentals: Periodic (month-to-month) and fixed-term (has a specified start and end date, e.g. a 12-month lease). More than a third of the NSW population rent their home. This number has increased by 17.6% since 2016, according to Government figures. Rentals have also become increasingly expensive over recent years. The median weekly cost to rent in NSW in March 2021 was $470. This figure jumped to $650 in March 2024. No fault evictions When a NSW tenant is notified that their landlord is ending a lease, they are typically given a move-out window between 30-90 days. No-grounds or ‘no fault’ evictions mean landlords can legally end a lease without providing a reason. The NSW Government has pledged to ban these evictions starting next year. Proposed changes Under the proposal, a landlord will be required to disclose the reason for ending a lease. This could include plans to sell, renovate or move into the property, or if a tenant has damaged a property or not paid rent. NSW Housing Minister, Rose Jackson, said the move will create a ”fairer” system for renters doing the right thing, “who should not have to be in a constant limbo with the possibility of an eviction for no reason just around the corner”.  The reforms will also extend the minimum notice period for renters on fixed-term leases of less than six months – from 30 to 60 days. Anyone on a six-month or longer lease will be given a minimum 90-day notice period of their lease ending, up from 60. NSW Premier Chris Minns said the reforms are intended to give “homeowners and renters more certainty”. “Bad tenants will still be able to be evicted. We don’t want homeowners to have to put up with bad behaviour,” he added. Elsewhere South Australia, Victoria, and the ACT have all banned no-fault evictions. Executive Director of advocacy group ‘Better Renting’, Joel Dignam, told TDA that since the ban was implemented in the ACT, there has been a 0.3% increase in the number of properties available for rent. Reaction NSW Opposition Leader Mark Speakman said the Coalition will consider the reforms, but argued the plan “will not solve” the state’s housing crisis. Vice President of the peak body ‘Property Owners NSW’ Debra Beck-Mawing described the proposal as “impractical,” and suggested the ban would lead to fewer available rentals in the state. In a statement to TDA, Beck-Mewing said: “The minute you buy a property you’re treated like a villain and an endless source of fees and taxes”. What now? NSW Opposition Leader Mark Speakman said the Coalition will consider the reforms, but argued the plan “will not solve” the state’s housing crisis. Vice President of the peak body ‘Property Owners NSW’ Debra Beck-Mawing described the proposal as “impractical,” and suggested the ban would lead to fewer available rentals in the state. In a statement to TDA, Beck-Mewing said: “The minute you buy a property you’re treated like a villain and an endless source of fees and taxes”."
demographic_info = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]  # Example one-hot encoded demographics
summary = summarize(article, demographic_info, model, tokenizer)
print(summary)