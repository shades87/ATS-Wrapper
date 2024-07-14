from pathlib import Path

def hashFunction(LLM, url, age, ed, nat, income, city):
    hashVal = str(hash(LLM+url+str(age)+str(ed)+str(nat)+str(income)+str(city)))
    return hashVal

def checkFileExists(hashVal):
    myFile = Path("summaries/"+hashVal+".txt")
    if myFile.is_file():
        return True
    
    else:
        return False
    
def writeSummary(hash, summary):
    f = open("summaries/"+hash +".txt", "w")
    f.write(summary)
    f.close()