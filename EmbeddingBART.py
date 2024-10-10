from torch.utils.data import DataLoader, Dataset
import torch
from dataset import load_data

import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers import BartTokenizer 
from transformers import Trainer, TrainingArguments, BartConfig
import torch.optim as optim

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#Abandoned this method after failure with embedding demographics into BERT
#May try again if BERT ends up working after I finish dissertation


device = torch.device('cuda')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    
data = load_data()

articles = [item['article'] for item in data]
summaries = [item['summary'] for item in data]
demographics = [{  # Create a demographics dict for each entry
    "age": item['age'],
    "ed": item['ed'],
    "nat": item['nat'],
    "metro": item['metro'],
    "income": item['income']
} for item in data]

df = pd.DataFrame(data)

encoder = OneHotEncoder(sparse_output=False)
demographic_features = df[['age', 'ed', 'nat', 'metro', 'income']]
demographic_embeddings = encoder.fit_transform(demographic_features)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
inputs = tokenizer(list(df['article']), return_tensors='pt', padding=True, truncation=True)

class BARTWithDemographics(BartForConditionalGeneration):
    def __init__(self, demographic_size, config, **kwargs):
        super().__init__(config, **kwargs)  # Pass config to the parent class
        self.demographic_size = demographic_size
        self.demographic_projection = torch.nn.Linear(demographic_size, 768)  # Adjust as necessary

    def forward(self, input_ids, attention_mask=None, demographics=None, **kwargs):
        # Get BART outputs
        bart_outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)

        # Project demographic features to match the BART hidden size
        demographic_embeddings = self.demographic_projection(demographics)

        # Combine BART outputs with demographic embeddings
        combined_embeddings = torch.cat((
            bart_outputs.last_hidden_state,
            demographic_embeddings.unsqueeze(1).repeat(1, bart_outputs.last_hidden_state.size(1), 1)
        ), dim=-1)

        return combined_embeddings
    
class SummarizationDataset(Dataset):
    def __init__(self, encodings, demographics, labels):
        self.encodings = encodings
        self.demographics = demographics
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'demographics': torch.tensor(self.demographics[idx], dtype=torch.float), 
            'labels': self.encodings['input_ids'][idx],  
        }

    def __len__(self):
        return len(self.labels)


train_dataset = SummarizationDataset(inputs, demographic_embeddings, list(df['summary']))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

config = BartConfig.from_pretrained('facebook/bart-base')
model = BARTWithDemographics(demographic_size=16, config=config)
model.to('cpu')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()


model.load_state_dict(torch.load('./bart_demographics_model'))

def generate_summary(article, demographics):

    inputs = tokenizer(article, return_tensors='pt', padding=True, truncation=True)
    demographics = torch.tensor(encoder.transform([demographics]).toarray(), dtype=torch.float)
    outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], demographics=demographics)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary

