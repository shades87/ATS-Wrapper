import sys
#While creating my BERT model this is a call to chat GPT 3.5 to have functionality to the web-ui
from openai import OpenAI #Chat GPT 3.5
#newspaper has a dependency lxml[html_clean] pip install lxml
from newspaper import Article #Get text from a News Article URL 
from newspaper import Config

import re #Regular Expression

#List of functions
#demographics() take a list of demographics represented by integers and turn it into instructions for a LLM
#Returns a string

#getArticle() Take a URL as an input and grab the text from that article
#Returns a string

#checkArticle() Check that a URL is from a list of news websites judged to be trusted
#Returns boolean


#create a function to change demographics into a string
#default for arguments is 0
def demographics(age=0, ed=0, income=0, region=0, country=0):
  demo = ""
  match age:
    case 1:
      demo = demo + "User is under 15. "

    case 2:
      demo = demo + "User is between 15 and 35. "

    case 3:
      demo = demo + "User is between 35 and 65. "

    case 4:
      demo = demo + "User is over 65. "

  match ed:
    case 1:
      demo = demo + "User has graduated high school. "
          
    case 2:
      demo = demo + "User has a Bachelor's degree. "

    case 3:
      demo = demo + "User has a PHD. "
      
  match country:
    case 1:
      demo = demo + "User is Australian. "

    case 2:
      demo = demo + "User is from New Zealand. "

    case 3:
      demo = demo + "User is from the UK. "

    case 4:
      demo = demo + "User is from the United States of America. "
          
  match region:
    case 1:
      demo = demo + "User lives in a city. "

    case 2:
      demo = demo + "User lives in a rural area. "

  match income:
    case 1:
      demo = demo + "User earns less than $30000 a year."
    
    case 2:
      demo = demo + "User earns between $30000 and $100000 a year."
        
    case 3:  
      demo = demo + "User earns over $100000 a year."
  
  if demo == "":
    demo = "Summarize the provided article"
  else:
    demo = "Summarize the provide article using language best suited to engage the user. " + demo
  
  return demo   

#Check that an article is from a recognized news source
def check_article(url):
  ok_list = ["abc.net.au/news/"]
  checking = True #boolean to
  url_regex = re.compile(r'https?://(?:www\.)?abc.net.au/news/[a-zA-Z0-9./]+')
  return bool(url_regex.match(url))


#create a function that can grab text from a website
def get_article_text(url):
  text = ""
  user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
  config = Config()
  config.browser_user_agent = user_agent

  article = Article(url)

  try:
    article.download()
    article.parse()
    print(article.text)
  except:
    print("Download failed")


  return text


#create a function to call chatgpt 3.5 using the demographics

def summarise(demo, cont):
  client = OpenAI()
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": demo},
      {"role": "user", "content": cont}
    ]
  )

  print(completion.choices[0].message)

#I think I should rearrange the functions
def flow(arr):
  if(arr[1] == 'checkURL'):
    check_article(arr[2])

  elif (arr[1] == 'Summarise'):
    article = get_article_text(arr[2])
    demos = demographics(arr[3],arr[4],arr[5],arr[6],arr[7])
    print(summarise(demos,article))
    sys.stdout.flush()

flow(sys.argv)