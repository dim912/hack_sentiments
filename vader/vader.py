from nltk.sentiment.vader import SentimentIntensityAnalyzer

#import nltk
#nltk.download('vader_lexicon')



sentence='My analysis is awesome'
sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores(sentence) 




# ------------------------------------ RESTFUL API ----------------------------

import json
from flask import Flask, jsonify, request,Response
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)

class Predictor(Resource):
    def post(self):
        json_data = request.get_json(force=True)

        result = []
        for tweetData in json_data:
            tweet = tweetData['tweet']
            date = tweetData['date']

            tweetSentiment = sid.polarity_scores(tweet) 
            if (tweetSentiment['pos'] > tweetSentiment['neg']):
                sentiment = 1;
            else:
                sentiment = 0;
            res = {"date":date, "tweet":tweet, "sentiment":sentiment}
            result.append(res)

        return Response(json.dumps(result),  mimetype='application/json')

api.add_resource(Predictor, '/predict/vader')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5446 ,debug=True)





