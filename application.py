from flask import Flask, request, json, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickle

ridge_model = pickle.load(open('Models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('Models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
        # get data from form
        Tempreture = float(request.form.get('Tempreture'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Clasess = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Tempreture, RH, Ws, Rain, FFMC, DMC, ISI, Clasess, Region]])

        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', results = result[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(debug=True)