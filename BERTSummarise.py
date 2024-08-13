import torch;
from oneHotEncodeTrainingBERT import *
from transformers import BertTokenizer

def bertSummarize(article, demographic_info, max_length=128):
    
    model_path = 'weights/demographic_bert_weights_two.pth'
    model = DemographicBERT(demographic_size=16)  # or the correct demographic size
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    model.to('cuda')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Tokenize the article
    
    input_ids = tokenizer.encode(article, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    
    # Prepare demographic tensor
    demographic_tensor = torch.tensor(demographic_info, dtype=torch.float).unsqueeze(0)
    
    # Move tensors to the appropriate device
    input_ids = input_ids.to("cuda")
    demographic_tensor = demographic_tensor.to("cuda")
    
    # Generate output
    with torch.no_grad():
        outputs = model(input_ids, demographic_tensor)
    
    # Get the summary
    summary_ids = torch.argmax(outputs, dim=-1)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary