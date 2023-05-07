import requests 
import json

response = requests.post("https://flaskdockerlogging-2t4hhd2ijq-uc.a.run.app/predict", data = json.dumps({'text': 'This is very bad'}), headers={
    'Content-Type':'application/json'
})

print(response.text)