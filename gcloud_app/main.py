import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
import requests

# UPLOAD_FOLDER = '../images/'
# ALLOWED_EXTENSIONS = {'png'}

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route("/", methods=['GET', 'POST'])
# def index():
#     def upload_file():
#         if request.method == 'POST':
#             if 'file1' not in request.files:
#                 return 'there is no file1 in form!'
#             file1 = request.files['file1']
#             path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
#             file1.save(path)
#             return path

app = Flask(__name__)


#check if file is already on disk, and if it is not, download from amazon S3
# loaded_model = pickle.load(open("random_forest.pkl", 'rb'))

#https://colin-week22-data-test.s3.us-east-2.amazonaws.com/random_forest.pkl
#requests.get of this url and saving to disk

if os.path.exists("random_forest.pkl"):
    print("file exists")

else:
    url = 'https://nu-final-project.s3.us-east-2.amazonaws.com/random_forest.pkl'
    r = requests.get(url, allow_redirects=True)
    open('random_forest.pkl', 'wb').write(r.content)
    
loaded_model = pickle.load(open("random_forest.pkl", 'rb'))


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,-1)
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/')
@app.route('/home', methods=['POST'])
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
        to_predict_list_dict = request.form.to_dict()
        to_predict_list_values = list(to_predict_list_dict.values())
        to_predict_list_float = list(map(float, to_predict_list_values))
        result = ValuePredictor(to_predict_list_float)
        prediction = str(result)
        return render_template("predict.html", prediction=prediction)

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

    


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404





