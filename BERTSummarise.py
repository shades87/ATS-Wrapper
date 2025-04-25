import os
from dotenv import load_dotenv
import torch;
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class DemographicBERT(nn.Module):
    def __init__(self, demographic_size):
        super(DemographicBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.demographic_embedding = nn.Linear(demographic_size, 768)
        self.decoder = nn.Linear(768, tokenizer.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input_ids, demographics):
        outputs = self.bert(input_ids)
        last_hidden_state = outputs.last_hidden_state
        demographic_embed = self.demographic_embedding(demographics)
        combined = last_hidden_state + demographic_embed.unsqueeze(1)
        decoded = self.decoder(combined)
        return self.softmax(decoded)



#model_path = 'weight/CNN4.pt'
model = DemographicBERT(demographic_size=16) 
#model.load_state_dict(torch.load(model_path))
load_dotenv()
token = os.getenv('HF_Token')

state_dic = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/shadyshadyshades/BERT-Demographics/upload/main/BERT_Demographics.bin",
    map_location="cpu",
    headers={"Authorization": f"Bearer {token}"}
)

model.load_state_dict(state_dic)

model.eval()  # Set the model to evaluation mode (I don't actually know what the default state is)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#torch.save(model.state_dict, "BERT_Demographics.bin")

def bertSummarize(article, demographic_info, max_length=128):
    
    print("Article inside bertSummaraize: " + article)
    
    # Tokenize the article, max length is 512 tokens which is BERT's max
    input_ids = tokenizer.encode(article, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    
    # Prepare demographic tensor
    demographic_tensor = torch.tensor(demographic_info, dtype=torch.float).unsqueeze(0)
    
    # Move tensors to the appropriate device
    input_ids = input_ids.to(device)
    demographic_tensor = demographic_tensor.to(device)
    
    # Generate summary
    with torch.no_grad():
        outputs = model(input_ids, demographic_tensor)
    
    # Get the summary
    summary_ids = torch.argmax(outputs, dim=-1)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary