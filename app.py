import pandas as pd
import numpy as np
import pickle
import uvicorn #ASGI
from fastapi import FastAPI
from BankNotes import BankNote

#Create app object
app = FastAPI()
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

#Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, User!!'}

#Route with single parameter, returns parameter within a message located 127.0.0.1:8000/AnyNameHere
@app.get('/{name')
def get_name(name: str):
    return {'message': f'Hello, {name}'}

#Expose prediction functionality make a prediction from the pass json data and return predicted bank note with confidence
@app.post('/predict')
def predict_NOTE(data:BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    #print(classifier.predict([[variance,skewness,curtosis,entropy]]))

    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])

    if prediction[0] > 0.5:
        prediction = "Fake note"

    else:
        prediction = "It's a Bank Note"

    return{
        'prediction': prediction
    }

#Run api with uvicorn
#run on 'http://127.0.0.1:8000'
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload