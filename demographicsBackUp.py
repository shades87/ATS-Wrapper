#While creating my BERT model this is a call to chat GPT 3.5 to have functionality to the web-ui

from openai import OpenAI
client = OpenAI()

#create a function to change demographics into a string
#default for arguments is 0
def demographics(age=0, ed=0, income=0, region=0, country=0):
      demo = ""

#create a function that can grab text from a website
def get_article_text(article):
  pass

#create a function to call chatgpt 3.5 using the demographics

def summarise(demo, cont):
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "Summarize the content using language to best engage someone whos is " + demo},
      {"role": "user", "content": cont}
    ]
  )

  print(completion.choices[0].message)