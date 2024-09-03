import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

df = pd.read_csv('CNN_Dataset/train.csv')

def process_data(df, max_length=512):
    input_ids = []
    attention_masks = []
    summary_ids = []
    demographics = []

    for _, row in df.iterrows():
        article = row['article']
        article_tokens = tokenizer.encode_plus(
            article,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(article_tokens['input_ids'])
        attention_masks.append(article_tokens['attention_mask'])

        summary = row['highlights']
        summar_tokens = tokenizer.encode_plus(
            summary,
            max_length = 128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        summary_ids.append(summar_tokens['input_ids'])

        demo = torch.zeros(16)
        demographics.append(demo)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    summary_ids = torch.cat(summary_ids, dim=0)
    demographics = torch.stack(demographics)

    return input_ids, attention_masks, demographics, summary_ids

input_ids, attention_mask, demographics, summary_ids = process_data(df)
torch.save((input_ids, attention_mask, demographics, summary_ids), 'data_tensors.pt')

def create_dataloader(input_ids, attention_mask, demographics, summary_ids, batch_size=2):
    dataset = TensorDataset(input_ids, attention_mask, demographics, summary_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

