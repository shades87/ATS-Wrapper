import sys
import os, glob
from openai import OpenAI #Chat GPT 3.5

from newspaper import Article 
from newspaper import Config

from demographicsBackUp import *
from hashFunctions import *
#Functions to create and load the dataset
#I intend to use synthetic data to train the BERT model
from demographicsBackUp import demographics

#Load the data in the .txt files in /dataset 
def data():
    data = []

    #example of what an item in data should look like
    #data = [
    #{"article": "Some news article text", "summary": "summary", "age": "Under 15", "ed": "PHD", "nat": "Australia", "metro": "Metro", "income": "Under 30K"}
    #]

    #For ever file in the dataset directory read and parse the .txt file and add it to the data object
    import os, glob
    path = 'dataset/'
    for filename in glob.glob(os.path.join(path, '*.txt')):
        with open(os.path.join(os.getcwd(), filename), 'r') as f: # open in readonly mode
            # do your stuff
            lines = f.readlines()
            url = ""
            article = ""
            summary = ""
            age = ""
            ed = ""
            nat = ""
            metro = ""
            income = ""

            for line in lines:
                line = line.rstrip()
                print(line)
                split = line.split(":") 
                print(split)
                match split[0]:
                    case "url":
                        url = split[1]

                    case "article":
                        article = split[1]

                    case "summary":
                        summary = split[1]
                    
                    case "age":
                        age = split[1]

                    case "ed":
                        ed = split[1]

                    case "nat":
                        nat = split[1]

                    case "metro":
                        metro = split[1]
                    
                    case "income":
                        income = split[1]
                
                f.close()
            data.append({"article": article, "summary": summary, "age": age, "ed": ed, "nat": nat, "metro": metro, "income": income})
            print(data[0])


#Programatically create data to summarise based on all possible demographic information
#Create the summaries using my chatgpt model
def create_data():

    #Grab the URL, create a summary for every possible demographic info
    #Every demographic info
    #Ed: 3 options, Age: 4 options, Nat: 4 options, Metro: 2 options, Income: 3 Options

    ed = 0;
    age = 0;
    nat = 0;
    metro = 0;
    income = 0;
    i = 1;
    index = 0;

    #Articles covering different topics from different sources
    articles = ["https://www.theguardian.com/australia-news/article/2024/jul/20/bad-actors-seizing-on-microsoft-it-outage-to-scam-public-clare-oneil-warns",
                "https://www.theguardian.com/sport/article/2024/jul/20/wallabies-australia-georgia-rugby-union-test-match-report",
                "https://thedailyaus.com.au/stories/heres-everything-we-know-about-project-2025/",
                "https://thedailyaus.com.au/stories/at-least-19-people-have-died-during-student-protests-in-bangladesh/",
                "https://www.afr.com/companies/financial-services/anz-s-toxic-trading-floor-roulette-spins-out-of-control-20240719-p5juwr",
                "https://www.afr.com/technology/what-is-crowdstrike-the-it-giant-behind-the-global-meltdown-20240719-p5jv4j",
                "https://www.abc.net.au/news/2024-07-20/wieambilla-christian-terrorist-train-brothers-abuse-claims/104114270",
                "https://www.abc.net.au/news/2024-07-20/paris-olympics-2024-diving-star-melissa-wu-returns-to-games/104121400",
                "https://www.busseltonmail.com.au/story/8697305/ironman-703-wa-hits-capacity-limited-entries-left-for-main-ironman/?cs=1435",
                "https://www.busseltonmail.com.au/story/8697067/geographe-nursery-gets-10k-grant-for-new-equipment/?cs=1435",
                "https://www.news.com.au/lifestyle/food/eat/7eleven-expands-fan-favourite-item-to-two-more-states/news-story/78e88a7a043ca917302c2927621b23e8",
                "https://www.news.com.au/lifestyle/food/eat/travel-influencer-slams-spanish-michelin-star-restaurant/news-story/c79117e964a2570466df93aa4cbead31",
                "https://7news.com.au/news/us-president-joe-bidens-team-defiant-as-more-democrats-say-to-step-aside-c-15413054",
                "https://7news.com.au/news/the-20k-that-colac-teen-noah-saved-while-couch-surfing-wasnt-enough-for-a-home-but-that-didnt-stop-him-c-15328771",
                "https://www.9news.com.au/national/bali-helicopter-crash-two-australians-three-indonesians-survive/324d78e8-0bf1-41e9-a02b-3f92fbbeb025",
                "https://www.9news.com.au/national/wind-and-rain-set-to-linger-for-days-across-australia/bbb2409c-6bab-4bef-a7c1-92839c5bb682",
                "https://www.sbs.com.au/news/article/six-million-times-faster-a-new-internet-speed-record-and-how-australia-compares/3ezzqj0ve",
                "https://www.sbs.com.au/news/article/top-un-court-rules-israeli-settlements-in-occupied-palestinian-territories-are-illegal/4wbcsd744",
                "https://www.sbs.com.au/news/article/cheryll-tried-to-get-pregnant-for-years-one-thing-sealed-her-decision-to-stop/dy8ig3rj7",
                "https://www.theguardian.com/technology/article/2024/jul/19/what-is-crowdstrike-microsoft-windows-outage",
                "https://edition.cnn.com/2024/07/19/politics/trump-rally-gunman-portrait-motive-invs/index.html",
                "https://edition.cnn.com/2024/07/19/business/starbucks-mobile-orders-third-place/index.html",
                "https://www.1news.co.nz/2024/07/20/hawkes-bay-community-bloody-angry-at-cuts-to-rural-school-bus-routes/",
                "https://www.1news.co.nz/2024/07/18/indira-stewart-the-battle-to-build-a-career-through-poverty/"]
    #1200 permutations
    articleURL = articles[index]

    for a in range(4):
        ed = a
        print("a: " + str(a))
        for b in range(5):
            age = b
            for c in range(5):
                nat = c
                for d in range(3):
                    metro = d
                    for e in range(4):
                        income = e
                        user = demographics(age, ed, income, metro, nat)
                        print(user)
                        #Every 50 summaries change the article to be summarised
                        if (i%50 == 0):
                            articleURL = articles[index]
                            #summarise and save

                            index = index + 1
                            
                        
                        #Increase the index, should go from 1 to 1200 (the number of permutations of the demographics)
                        #Every 50 summaries swap articles to get a range of summaries
                        i = i + 1
                        


                        #function to summarise and save the data

def save_summary(article, summary, url, ed, nat, income, age, metro):
    hash_file_name = hashFunction("ChatGPT", article, age, ed, nat, metro, income)

    text_file = open( "dataset/"+hash_file_name+".txt","w")
    text_file.write("url:"+url+"\n")
    text_file.write("article:"+article+"\n")
    text_file.write("summary:"+summary+"\n")
    text_file.write("age:"+age+"\n")
    text_file.write("ed:"+ed+"\n")
    text_file.write("nat:"+nat+"\n")
    text_file.write("metro:"+metro+"\n")
    text_file.write("income:"+income+"\n")
    text_file.close()


create_data()