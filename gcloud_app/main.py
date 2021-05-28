import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

#check if file is already on disk, and if it is not, download from amazon S3
loaded_model = pickle.load(open("random_forest.pkl", 'rb'))

#https://colin-week22-data-test.s3.us-east-2.amazonaws.com/random_forest.pkl
#requests.get of this url and saving to disk

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,-1)
    
    result = loaded_model.predict(to_predict)
    proba = loaded_model.predict_proba(to_predict)
    print(result)
    print(proba)
    print(loaded_model.classes_)
    print(to_predict)
    return result[0]

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        prediction = str(result)
        return render_template("predict.html", prediction=prediction)

    


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404




if __name__ == '__main__':
    app.run(debug=True)

