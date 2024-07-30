import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import torch.optim as optim
from dataset import *

#I asked ChatGPT for help using One Hot Encoding to train BERT and used the results to help build this 
torch.cuda.empty_cache()
#set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset that I've created
data = load_data()

df = pd.DataFrame(data)

# One-hot encode demographic information
demographic_fields = ['age', 'ed', 'nat', 'income', 'metro']
one_hot_encoded_demographics = pd.get_dummies(df[demographic_fields], drop_first=True)

# Combine the original dataframe with the one-hot encoded demographic data
df = pd.concat([df, one_hot_encoded_demographics], axis=1)
df = df.drop(columns=demographic_fields)

#check the dataframe to see if one hot encoding worked
print("head")
print(df.head())

print("Verify length of one hot encoding")
# Verify the number of one-hot encoded fields
num_demographic_fields = len(one_hot_encoded_demographics.columns)
print(f"Number of one-hot encoded demographic fields: {num_demographic_fields}")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("vocab size: " + str(tokenizer.vocab_size))

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
        demographic_info = row[-16:].values.astype(float)  # Adjust the slicing as per your dataframe structure
        
        # Print shapes for debugging
        print(f"Shape of article_tokens: {len(article_tokens)}")
        print(f"Shape of summary_tokens: {len(summary_tokens)}")
        print(f"Shape of demographic_info: {demographic_info.shape}")

        return {
            'input_ids': torch.tensor(article_tokens, dtype=torch.long),
            'summary_ids': torch.tensor(summary_tokens, dtype=torch.long),
            'demographics': torch.tensor(demographic_info, dtype=torch.float)
        }

# Prepare dataset and dataloader
dataset = NewsDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model, loss, and optimizer
model = DemographicBERT(demographic_size=16).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        summary_ids = batch['summary_ids'].to(device)
        demographics = batch['demographics'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, demographics)
        
        # Check shapes before reshaping
        print(f"Shape of outputs before reshaping: {outputs.shape}")
        print(f"Shape of summary_ids before reshaping: {summary_ids.shape}")
        
        # Reshape outputs and summary_ids
        outputs = outputs.view(-1, tokenizer.vocab_size)
        summary_ids = summary_ids.view(-1)
        
        # Check shapes after reshaping
        print(f"Shape of outputs after reshaping: {outputs.shape}")
        print(f"Shape of summary_ids after reshaping: {summary_ids.shape}")
        
        # Ensure that the reshaped tensors have matching sizes
        #assert outputs.shape[0] == summary_ids.shape[0], \
            #f"Mismatch: outputs {outputs.shape[0]} vs summary_ids {summary_ids.shape[0]}"
        
        # Compute loss
        loss = criterion(outputs, summary_ids)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Free up unused memory
torch.cuda.empty_cache()
# Load the model if needed
#model = DemographicBERT(demographic_size=len(one_hot_encoded_demographics.columns))
#model.load_state_dict(torch.load(model_path))
#model.eval()
#print("Model loaded successfully")

# Inference function
def summarize(article, demographic_info, model, tokenizer):
    input_ids = tokenizer.encode(article, return_tensors='pt')
    demographic_tensor = torch.tensor(demographic_info, dtype=torch.float).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids, demographic_tensor)
    summary_ids = torch.argmax(outputs, dim=-1)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary



# Example usage, let's check
article = "SW will ban ‘no fault’ evictions, bringing the state in line with other parts of the country, including Victoria. Landlords currently have the power to end a lease without giving their tenant a reason. However, the NSW Government will introduce legislation to ban this practice. Notice periods for evictions will also be extended under the plan. The State Government said the reforms are set to be rolled out “early next year”. Renting in NSW There are two types of tenancy agreements (leases) for NSW rentals: Periodic (month-to-month) and fixed-term (has a specified start and end date, e.g. a 12-month lease). More than a third of the NSW population rent their home. This number has increased by 17.6% since 2016, according to Government figures. Rentals have also become increasingly expensive over recent years. The median weekly cost to rent in NSW in March 2021 was $470. This figure jumped to $650 in March 2024. No fault evictions When a NSW tenant is notified that their landlord is ending a lease, they are typically given a move-out window between 30-90 days. No-grounds or ‘no fault’ evictions mean landlords can legally end a lease without providing a reason. The NSW Government has pledged to ban these evictions starting next year. Proposed changes Under the proposal, a landlord will be required to disclose the reason for ending a lease. This could include plans to sell, renovate or move into the property, or if a tenant has damaged a property or not paid rent. NSW Housing Minister, Rose Jackson, said the move will create a ”fairer” system for renters doing the right thing, “who should not have to be in a constant limbo with the possibility of an eviction for no reason just around the corner”.  The reforms will also extend the minimum notice period for renters on fixed-term leases of less than six months – from 30 to 60 days. Anyone on a six-month or longer lease will be given a minimum 90-day notice period of their lease ending, up from 60. NSW Premier Chris Minns said the reforms are intended to give “homeowners and renters more certainty”. “Bad tenants will still be able to be evicted. We don’t want homeowners to have to put up with bad behaviour,” he added. Elsewhere South Australia, Victoria, and the ACT have all banned no-fault evictions. Executive Director of advocacy group ‘Better Renting’, Joel Dignam, told TDA that since the ban was implemented in the ACT, there has been a 0.3% increase in the number of properties available for rent. Reaction NSW Opposition Leader Mark Speakman said the Coalition will consider the reforms, but argued the plan “will not solve” the state’s housing crisis. Vice President of the peak body ‘Property Owners NSW’ Debra Beck-Mawing described the proposal as “impractical,” and suggested the ban would lead to fewer available rentals in the state. In a statement to TDA, Beck-Mewing said: “The minute you buy a property you’re treated like a villain and an endless source of fees and taxes”. What now? NSW Opposition Leader Mark Speakman said the Coalition will consider the reforms, but argued the plan “will not solve” the state’s housing crisis. Vice President of the peak body ‘Property Owners NSW’ Debra Beck-Mawing described the proposal as “impractical,” and suggested the ban would lead to fewer available rentals in the state. In a statement to TDA, Beck-Mewing said: “The minute you buy a property you’re treated like a villain and an endless source of fees and taxes”."
demographic_info = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]  # Example one-hot encoded demographics
summary = summarize(article, demographic_info, model, tokenizer)
print(summary)