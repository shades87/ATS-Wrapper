from dataset import *
import pandas as pd
from transformers import BertTokenizer

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.nn as nn
from transformers import BertModel

import torch.optim as optim

import time

#BERT Model written with the assistance of chatGPT 4o

data = load_data()
print("length of data: " + str(len(data)))

print(data[0])

# Load your data into a DataFrame
df = pd.DataFrame(data)
print("Original dataframe shape:", df.shape)
print("Number of duplicate rows:", df.duplicated().sum())
# One-hot encode demographic information
demographic_fields = ['age', 'ed', 'nat', 'income', 'metro']
one_hot_encoded_demographics = pd.get_dummies(df[demographic_fields], drop_first=True)

print(one_hot_encoded_demographics.shape)
print(data[0])
print(one_hot_encoded_demographics.iloc[0])

print(data[500])
print(one_hot_encoded_demographics.iloc[500])
# Combine the original dataframe with the one-hot encoded demographic data
df = pd.concat([df, one_hot_encoded_demographics], axis=1)
df = df.drop(columns=demographic_fields)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print(df.shape)
print(df.columns)

def tokenize_data(df):
    df['article_tokens'] = df['article'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=512))
    df['summary_tokens'] = df['summary'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=128))
    return df

df = tokenize_data(df)

class NewsDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        article_tokens = torch.tensor(row['article_tokens'], dtype=torch.long)
        summary_tokens = torch.tensor(row['summary_tokens'], dtype=torch.long)
        
        #Limit article to 512 tokens which is BERT's max, if needed
        #article_tokens = self.tokenizer.encode(row['article'], truncation=True, padding='max_length', max_length=512)

        #debug print the values in the rows
        #for i, value in enumerate(row.values):
            #print(f"Index {i}: Value {value} (Type: {type(value)})")
        #Exclude the article and summary for demogrpahic info
        demographic_info = torch.tensor(row[2:18].values.astype(float), dtype=torch.float)  # Adjust the slicing as per your dataframe structure
        
        return {
            'input_ids': article_tokens,
            'summary_ids': summary_tokens,
            'demographics': demographic_info
        }
    
dataset = NewsDataset(df)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

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
    

model = DemographicBERT(demographic_size=16).to("cuda")
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)


#check that model is running on nvidia gpu
print(next(model.parameters()).device)
print(torch.cuda.is_available())  # Should be True if CUDA is available
print(torch.cuda.current_device())  # Prints the current device index
print(torch.cuda.get_device_name(0))  # Prints the name of the device

def train():
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to("cuda")
            summary_ids = batch['summary_ids'].to("cuda")
            demographics = batch['demographics'].to("cuda")
            
            optimizer.zero_grad()
            outputs = model(input_ids, demographics)

            #debug check size of outputs and summary_ids
            #print(f"Original shape of outputs: {outputs.shape}")  # Should be [batch_size, sequence_length, vocab_size]
            #print(f"Original shape of summary_ids: {summary_ids.shape}")  # Should be [batch_size, sequence_length]


            #truncate the outputs -> does this affect the summaries though?
            outputs = outputs[:, :128, :]  # Truncate to match summary length

            # Flatten outputs to [batch_size * summary_length, vocab_size]
            outputs = outputs.reshape(-1, tokenizer.vocab_size)  # Now [256, 30522]

            # Flatten summary_ids to [batch_size * sequence_length]
            summary_ids = summary_ids.view(-1)

            # Check shapes after reshaping
            #print(f"Shape of outputs after reshaping: {outputs.shape}")
            #print(f"Shape of summary_ids after reshaping: {summary_ids.shape}")

            
            loss = criterion(outputs, summary_ids)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

start_time = time.time()
train()
end_time = time.time()

# Total time taken
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")
# Save the model weights
torch.save(model.state_dict(), 'weights/demographic_bert_weights_two.pth')

# Load the model weights
#model.load_state_dict(torch.load('weights/demographic_bert_weights_two.pth'))
#model.eval()

def summarize(article, demographic_info, model, tokenizer):
    maxSize = 512
    input_ids = tokenizer.encode(article, return_tensors='pt', truncation=True, max_length=maxSize).to("cuda")
    demographic_tensor = torch.tensor(demographic_info, dtype=torch.float).unsqueeze(0).to("cuda")
    with torch.no_grad():
        outputs = model(input_ids, demographic_tensor)
    summary_ids = torch.argmax(outputs, dim=-1)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

#Example usage
article = "Dire financial straits are leading droves of Olympic athletes to sell images of their bodies to subscribers on OnlyFans — known for sexually explicit content — to sustain their dreams of gold at the Games. As they struggle to make ends meet, a spotlight is being cast on an Olympics funding system that watchdog groups condemn as \"broken,\" claiming most athletes \"can barely pay their rent\". The Olympics, the world's biggest sporting stage, bring in billions of dollars in TV rights, ticket sales and sponsorship, but most athletes must fend for themselves financially. The International Olympic Committee (IOC) did not express concern about the situation. When asked about athletes turning to OnlyFans, IOC spokesman Mark Adams said, \"I would assume that athletes, like all citizens, are allowed to do what they can.\" Watching his sponsorships dry up and facing mounting costs, Team GB's Jack Laugher was among the pantheon of Olympic athletes using the often-controversial platform to get to the Games — or simply survive. After medalling at the Tokyo Olympics in 2021, Laugher, who scored another bronze in Paris last week for Great Britain, said he was waiting for funding that never materialised. His OnlyFans account, costing $10 a month for a subscription, says he posts \"SFW (safe for work) content in Speedos, briefs, boxers\". A recent post from the Olympics got more than 1,400 likes. \"For me, it's been an absolute lifeline,\" he said, before he was whisked away mid-interview by a British team official, underscoring the sensitivity of the issue. Multiple current and former Olympians were asked about the situation, painting a sobering portrait of what they had to do — and bare — to get to Paris. Laugher, and other past and present Olympians — including Australian diver Matthew Mitcham, New Zealand rower Robbie Manson, Canadian pole vaulter Alysha Newman, and divers Timo Barthel and Diego Belleza Isaias — found a measure of financial stability in OnlyFans that other funding failed to provide. Unable to secure traditional sponsorships, Mitcham began posting photos on OnlyFans, including semi-frontal nudes, earning triple the amount he received as a top athlete. \"That body is an amazing commodity that people want to pay to see. It's a privilege to see a body that has six hours of work every day, six days a week put into it to make it Adonis-like,\" said Mitcham, who describes himself as a \"sex worker-lite\". Manson, meanwhile, credited OnlyFans with boosting his athletic performance, saying his content included \"thirst traps,\" but nothing pornographic. \"My content is nude or implied nude. I keep it artistic, I have fun with it and try not to take myself too seriously,\" he said. \"That's something I've also tried to maintain in my approach to rowing … this approach has helped me achieve a personal best result at the Olympics.\" While some athletes say they don't see what they're doing as sex work, German diver Barthel put it frankly: \"In sport, you wear nothing but a Speedo, so you're close to being naked.\" Global Athlete, an organisation created by athletes to address the power imbalance in sports, decried the dire state of Olympic financing. \"The entire funding model for Olympic sport is broken. The IOC generates now over US$1.7 billion ($2.6 billion) per year and they refuse to pay athletes who attend the Olympics,\" said Rob Koehler, Global Athlete's director general. He criticised the IOC for forcing athletes to sign away their image rights. \"The majority of athletes can barely pay their rent, yet the IOC, national Olympic committees and national federations that oversee the sport have employees making over six figures. They all are making money off the backs of athletes,\" he said. \"In a way, it is akin to modern-day slavery.\" Multiple athletes who were asked about their financial situations confirmed they have had to pay their way to the Olympics.  While stars like Michael Phelps and Simone Biles can make millions, most athletes struggle to cover the cost of competing on the global stage. These can include coaching, physical therapy and equipment, at a cost of thousands of dollars a month, as well as basic living expenses.  Some delegations fund training, with the athletes covering medical bills and daily expenses. In other delegations, athletes pay for everything themselves. Olympic athletes are generally given just one or two tickets for friends and family, obliging them to pay for additional tickets so their loved ones can attend their events. \"The IOC tries to convince these athletes that their lives will change after becoming an Olympian — there is nothing further from the truth. The fact is the majority of athletes are left in debt, face depression, and they are lost once finishing sport with no future employment pathway,\" Koehler said. Pole vaulter Alysha Newman has used the money she earned from OnlyFans to buy property and build up her savings. \"I never loved how amateur athletes can never make a lot of money,\" she said.  \"This is where my entrepreneurial skills came in.\" Adams, the IOC spokesman, said at a press conference he wasn't aware of the trend and dismissed concern about the subject.  The AP requested details from the IOC on how it helps athletes financially, and the IOC referred the AP to a swathe of links with scant detail, without elaborating or providing further comment.  A statement from the IOC executive board said the IOC distributes 90 per cent of its revenues to \"the development of sport and athletes,\" but didn't go into detail. OnlyFans has expressed solidarity for its athletes. \"OnlyFans is helping them to support training and living costs, and providing the tools for success on and off the field,\" the platform said in a statement. It highlights other \"exceptionally talented OnlyFans athlete creators who were unable to compete in Paris this year,\" including British divers Matthew Dixon, Daniel Goodfellow, and Matty Lee, along with British speed skater Elise Christie and Spanish fencer Yulen Pereira. Athletes on OnlyFans say they have been forced to grapple with societal stigma.  Some said they had been asked if they were now porn stars, and one diver's profile even clarified: \"I'm a Team GB (Great Britain) diver, not a porn star.\" But others like Mitcham have been vocal about their experiences. \"Some people are judgy about sex work. People say it's a shame or even that it is shameful,\" Mitcham said.  \"But what I do is a very light version of sex work, like the low-fat version of mayonnaise … selling the sizzle rather than the steak.\" Mexican diver Diego Balleza Isaias, however, said the experience left him feeling dejected. Balleza Isaias said he joined OnlyFans in 2023 to get to the Olympics and support his family. After failing to qualify for Paris, he planned to close his account. \"I firmly believe that no athlete does this because they like it,\" he said. \"It's always going to be because you need to.\" The financial incentive can be considerable. French pole vaulter Anthony Ammirati shot to unexpected fame when his genitals snagged on the bar at a qualifying event. According to reports, an adult site then offered him a six-figure sum to showcase his \"talent\" on its platform. Mitcham suggested OnlyFans was superior to GoFundMe, as athletes aren't just asking for money or \"handouts\". \"With OnlyFans, athletes are actually providing a product or service, something of value for the money they're receiving,\" he explained, emphasising the need to reframe thinking. \"It's making athletes entrepreneurs.\""
demographic_info = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]  # Example one-hot encoded demographics
summary = summarize(article, demographic_info, model, tokenizer)
print(summary)