def hashFunction(LLM, url, age, ed, nat, income, city):
    hash = str(hash(LLM+url+age+ed+nat+income+city))
    return hash

def checkFileExists(hash):
    myFile = "summaries/"+hash+".txt"
    if myFile.is_file():
        return True
    
    else:
        return False
    
def writeSummary(hash, summary):
    f = open("summaries/"+hash +".txt", "w")
    f.write(summary)
    f.close()