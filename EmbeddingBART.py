from torch.utils.data import DataLoader, Dataset
import torch
from dataset import loadForBartEmbedding

import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers import BartTokenizer
import torch.optim as optim


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

class SummarizationDataset(Dataset):
    def __init__(self, articles, summaries, indices, tokenizer, device='cpu'):
        self.articles = articles
        self.summaries = summaries
        self.indices = indices
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        # Get the article and summary at the specified index
        article_text = self.articles[idx]
        summary_text = self.summaries[idx]
        sample_indices = self.indices[idx] 
        
        article_tokens = self.tokenizer.encode(article_text, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(self.device)
        summary_tokens = self.tokenizer.encode(summary_text, return_tensors='pt', truncation=True, padding='max_length', max_length=150).to(self.device)

        # Get the demographic indices for this sample
        age_idx = torch.tensor(sample_indices['age_index']).to(self.device)
        education_idx = torch.tensor(sample_indices['ed_index']).to(self.device)
        nationality_idx = torch.tensor(sample_indices['nationality_index']).to(self.device)
        locale_idx = torch.tensor(sample_indices['locale_index']).to(self.device)
        income_idx = torch.tensor(sample_indices['income_index']).to(self.device)
        # Return the data as a dictionary
        return {
            'input_ids': torch.tensor(article_tokens),
            'decoder_input_ids': torch.tensor(summary_tokens),
            'age_idx': torch.tensor(age_idx),
            'education_idx': torch.tensor(education_idx),
            'nationality_idx': torch.tensor(nationality_idx),
            'locale_idx': torch.tensor(locale_idx),
            'income_idx': torch.tensor(income_idx)
        }
    
data = loadForBartEmbedding()

articles = data["articles"]
summaries = data["summaries"]
indices = data["demographics"]

dataset = SummarizationDataset(articles, summaries, indices, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

class BartWithDemographics(nn.Module):
    def __init__(self):
        super(BartWithDemographics, self).__init__()
        self.bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Embedding layers for demographics
        self.age_embedding = nn.Embedding(5, 128)  # 5 categories for age, embedding size 128
        self.education_embedding = nn.Embedding(4, 128)  # 4 categories for education
        self.nationality_embedding = nn.Embedding(4, 128)  # 4 categories for nationality
        self.locale_embedding = nn.Embedding(2, 128)  # 2 categories for locale
        self.income_embedding = nn.Embedding(3, 128)  # 3 categories for income
        
    def forward(self, input_ids, decoder_input_ids, age_idx, education_idx, nationality_idx, locale_idx, income_idx):
        # Get BART's outputs
        bart_outputs = self.bart_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        
        # Get embeddings for each demographic feature
        age_embeds = self.age_embedding(age_idx).to(self.device)
        education_embeds = self.education_embedding(education_idx).to(self.device)
        nationality_embeds = self.nationality_embedding(nationality_idx).to(self.device)
        locale_embeds = self.locale_embedding(locale_idx).to(self.device)
        income_embeds = self.income_embedding(income_idx).to(self.device)
        
        input_ids = input_ids.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        # Concatenate demographic embeddings (could also sum, depending on preference)
        demographic_embeds = torch.cat([age_embeds, education_embeds, nationality_embeds, locale_embeds, income_embeds], dim=-1)
        
        # You can now combine demographic embeddings with the output from BART
        combined_output = bart_outputs.logits + demographic_embeds  # This is just one way, adjust as needed
        
        return combined_output
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BartWithDemographics().to(device)
model.train()

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(1):
    for batch in dataloader:

        input_ids = batch['input_ids'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        age_idx = batch['age_idx'].to(device)
        education_idx = batch['education_idx'].to(device)
        nationality_idx = batch['nationality_idx'].to(device)
        locale_idx = batch['locale_idx'].to(device)
        income_idx = batch['income_idx'].to(device)

        print(f"input_ids device: {input_ids.device}")
        print(f"decoder_input_ids device: {decoder_input_ids.device}")
        print(f"age_idx device: {age_idx.device}")
        print(f"education_idx device: {education_idx.device}")
        print(f"nationality_idx device: {nationality_idx.device}")
        print(f"locale_idx device: {locale_idx.device}")
        print(f"income_idx device: {income_idx.device}")
        print(f"Model device: {next(model.parameters()).device}")
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            decoder_input_ids=batch['decoder_input_ids'],
            age_idx=batch['age_idx'],
            education_idx=batch['education_idx'],
            nationality_idx=batch['nationality_idx'],
            locale_idx=batch['locale_idx'],
            income_idx=batch['income_idx']
        )
        
        # Compute loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), batch['decoder_input_ids'].view(-1))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    model_save_path = "bart_with_demographics_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)


def generate_summary(model, tokenizer, article, demographics):
    # Tokenize the input article
    input_ids = tokenizer.encode(article, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Prepare the demographic indices
    age_idx = torch.tensor([demographics["age"]]).unsqueeze(0).to(device)  # Shape: (1, 1)
    education_idx = torch.tensor([demographics["education"]]).unsqueeze(0).to(device)
    nationality_idx = torch.tensor([demographics["nationality"]]).unsqueeze(0).to(device)
    locale_idx = torch.tensor([demographics["locale"]]).unsqueeze(0).to(device)
    income_idx = torch.tensor([demographics["income"]]).unsqueeze(0).to(device)
    
    # Generate a summary using the model
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=input_ids,
            age_idx=age_idx,
            education_idx=education_idx,
            nationality_idx=nationality_idx,
            locale_idx=locale_idx,
            income_idx=income_idx,
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Test article for summarization
test_article = "The economy is expected to grow at a steady pace over the next few years, with an increase in employment opportunities and technological advancements."

# Define two sets of demographics for testing
demographics_1 = {"age": 1, "education": 2, "nationality": 0, "locale": 1, "income": 2}
demographics_2 = {"age": 3, "education": 1, "nationality": 2, "locale": 0, "income": 3}

# Generate summaries
summary_1 = generate_summary(model, tokenizer, test_article, demographics_1)
summary_2 = generate_summary(model, tokenizer, test_article, demographics_2)

# Print the results
print("Summary for Demographics 1:")
print(summary_1)
print("\nSummary for Demographics 2:")
print(summary_2)