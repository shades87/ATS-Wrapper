from pathlib import Path
import hashlib

def hashFunction(LLM, url, age, ed, nat, income, city):
    m = hashlib.sha256()
    m.update(str((LLM+url+str(age)+str(ed)+str(nat)+str(income)+str(city))).encode())
    m.digest()
    hashVal = m.hexdigest()
    
    return hashVal

def checkFileExists(hashVal):
    myFile = Path("summaries/"+hashVal+".txt")
    if myFile.is_file():
        return True
    
    else:
        return False
    
def writeSummary(hash, summary):
    f = open("summaries/"+hash +".txt", "w")
    f.write(str(summary))
    f.close()

def readFile(hash):
    f = open("summaries/"+hash +".txt", "r")

    s = f.read()
    f.close()
    return {"content:" + s}
