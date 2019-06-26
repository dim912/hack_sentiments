import numpy as np

wordsList = np.load('wordsList.npy') #co occurrence dictionary
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]

####################################

from os import listdir
from os.path import isfile, join


# loard all positive and negative file names to arrys
positiveFiles = ['txt_sentoken/pos/' + f for f in listdir('txt_sentoken/pos/') if isfile(join('txt_sentoken/pos/', f))]
negativeFiles = ['txt_sentoken/neg/' + f for f in listdir('txt_sentoken/neg/') if isfile(join('txt_sentoken/neg/', f))]
print(positiveFiles[0])


#load number of words in each file to numWords array

numWords = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        lines=f.readlines()
        counter = 0
        for line in lines:
            counter += len(line.split())
        numWords.append(counter)
print('Positive files finished')

#load number words in in each file to  the same array
for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        lines=f.readlines()
        counter = 0
        for line in lines:
            counter += len(line.split())
        numWords.append(counter)
print('Negative files finished')

numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords)/len(numWords))


# -----------------clean lines and get only the alphabetical characters--------------------
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


######################################

maxSeqLength = 750 #upper limit for the number of words that are considerred in a given file

#two dimensional arry containing fileNumber and word number as the weight as per wordsList is the value
ids = np.zeros((numFiles, maxSeqLength), dtype='int32')

fileCounter = 0

for pf in positiveFiles:
    with open(pf, "r") as f:
        indexCounter = 0
        lines=f.readlines()
        for line in lines:
            cleanedLine = cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    ids[fileCounter][indexCounter] = wordsList.index(word)  #returns the position of the word in the dictionary
                except ValueError:
                    ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
                print(fileCounter,indexCounter)
                indexCounter = indexCounter + 1
                if indexCounter >= maxSeqLength:
                    break
            if indexCounter >= maxSeqLength:
                    break
        fileCounter = fileCounter + 1

for nf in negativeFiles:
    with open(nf, "r") as f:
        indexCounter = 0
        lines=f.readlines()
        for line in lines:
            cleanedLine = cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    ids[fileCounter][indexCounter] = wordsList.index(word)
                except ValueError:
                    ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
                indexCounter = indexCounter + 1
                if indexCounter >= maxSeqLength:
                    break
            if indexCounter >= maxSeqLength:
                    break
        fileCounter = fileCounter + 1

# idsMatrix2 is the numeric representation of the training data. It contains both negative and positive samples.
# the dimension of ids is ( number of files * 750=maxSeqLength)
# id metrix is called as co-occurrence matrix

np.save('idsMatrix2', ids)