#While creating my BERT model this is a call to chat GPT 3.5 to have functionality to the web-ui

from openai import OpenAI

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
            

        

#create a function that can grab text from a website
def get_article_text(article):
  pass

#create a function to call chatgpt 3.5 using the demographics

def summarise(demo, cont):
  client = OpenAI()
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "Summarize the content using language to best engage someone whos is " + demo},
      {"role": "user", "content": cont}
    ]
  )

  print(completion.choices[0].message)