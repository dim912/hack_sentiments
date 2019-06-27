numDimensions = 300
maxSeqLength = 750
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000



import numpy as np
wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')



import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


sess = tf.InteractiveSession()
saver = tf.train.Saver()


saver = tf.train.import_meta_graph('models/pretrained_lstm.ckpt-60000.meta')
saver.restore(sess,tf.train.latest_checkpoint('models'))

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
    split = cleanedSentence.split()
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 399999 #Vector for unkown words
    return sentenceMatrix


inputText = '''New #macOS malware found exploiting the latest GateKeeper bypass vulnerability that was disclosed publicly last month after #Apple left it unpatched for 90 days'''

inputMatrix = getSentenceMatrix(inputText)



predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]

print(predictedSentiment)


if (predictedSentiment[0] > predictedSentiment[1]):
    print("Positive Sentiment")
else:
    print("Negative Sentiment")


secondInputText = '''When I was a kid I watched this many times over,
and I remember whistling the "Happy Cat" song quite often.
All the songs are great, and actually meorable, unlike many children's musicals,
where the songs are just stuck in for no real reason. The scenes and costumes are lavish,
and the acting is very well-done, which isn't surprising, considering the cast.
I'd recommend this film to children and parents alike, who love magic and fairytales.
And it actually IS a movie you can watch together, as it won't drive adults up the wall.'''

secondInputMatrix = getSentenceMatrix(secondInputText)



predictedSentiment = sess.run(prediction, {input_data: secondInputMatrix})[0]
if (predictedSentiment[0] > predictedSentiment[1]):
    print("Positive Sentiment")
else:
    print("Negative Sentiment")


# ------------------------------------ RESTFUL API ----------------------------

import json
from flask import Flask, jsonify, request,Response
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)



from flask_cors import CORS, cross_origin
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'




class Predictor(Resource):
   
    @cross_origin()
    def post(self):
        json_data = request.get_json(force=True)

        result = []
        for tweetData in json_data:
            tweet = tweetData['tweet']
            date = tweetData['date']

            tweetMatrix = getSentenceMatrix(tweet)
            tweetSentiment = sess.run(prediction, {input_data: tweetMatrix})[0]
            if (tweetSentiment[0] > tweetSentiment[1]):
                sentiment = 1;
            else:
                sentiment = 0;
            res = {"date":date, "tweet":tweet, "sentiment":sentiment}
            result.append(res)

        return Response(json.dumps(result),  mimetype='application/json')

api.add_resource(Predictor, '/predict/tf')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5444 ,debug=True)



