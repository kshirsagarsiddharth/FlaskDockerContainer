from flask import Flask, jsonify, request
from utilities import predict_pipeline

app = Flask(__name__)

import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')


@app.route("/predict", methods=["POST","GET"])
def predict():
    if request.method == "POST":
        data_json = request.json
        try:
            sample = data_json["text"]
        except KeyError:
            return jsonify({"error": "No text sent"})
        
        sample = [sample] 
        predictions = predict_pipeline(sample) 

        try: 
            result = jsonify(predictions) 
        except TypeError as e: 
            result = jsonify({'error': str(e)})
        
        return result 
    else: 
        return "Welcome"

if __name__ == "__main__": 
    app.run(host = "0.0.0.0", debug=True)
    
