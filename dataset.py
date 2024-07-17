def data():
    data = [
    {"article": "Some news article text", "summary": "summary", "age_group": "18-24", "gender": "male", "education": "college", "region": "north", "interests": "technology"},
    {"article": "Another article", "summary": "Another summary", "age_group": "25-34", "gender": "female", "education": "high_school", "region": "south", "interests": "sports"},
    # I need more data entries...
    ]

    return data

#Create data to summarise based on all possible demographic information
def create_data():
    articleURL = ""
    ed = ["Under 15", ]