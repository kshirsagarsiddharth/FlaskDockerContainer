import requests 
import json

response = requests.post("http://localhost:5000/predict", data = json.dumps({'text': 'This is very bad'}), headers={
    'Content-Type':'application/json'
})

print(response.text)