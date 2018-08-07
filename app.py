# Imports
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)

import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the csv file into a pandas DataFrame
aba = pd.read_csv('abalone.data')

X = aba[["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]]
y = aba["rings"].values.reshape(-1, 1)

from sklearn.model_selection import train_test_split

X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

from sklearn.metrics import mean_squared_error

predictions = model.predict(X_test_scaled)
MSE = mean_squared_error(y_test_scaled, predictions)
r2 = model.score(X_test_scaled, y_test_scaled)

rings = 0
age = 0

# Flask  Setup and Routes
app = Flask(__name__)

# Create route that renders index.html template
@app.route("/")
def index():
    return render_template("index.html", 
        mse=MSE, 
        r2=r2, 
        length=0.455,
        diameter = 0.365,
        height = 0.095,
        whole_weight = 0.514,
        shucked_weight = 0.2245,
        viscera_weight = 0.101,
        shell_weight = 0.15,
        sex_male = 'selected',
        sex_female = '',
        sex_infant = '',
        rings = rings, 
        age = age)

@app.route('/calc', methods=['POST'])
def calc():
    sex_male = 0
    sex_female = 0
    sex_infant = 0
    sex_male_selected = ''
    sex_female_selected = ''
    sex_infant_selected = ''
    if request.form['sex'] == 'M':
        sex_male = 1
        sex_male_selected = 'selected'
    elif request.form['sex'] == 'F':
        sex_female = 1
        sex_female_selected = 'selected'
    elif request.form['sex'] == 'I':
        sex_infant = 1
        sex_infant_selected = 'selected'

    rings_a = y_scaler.inverse_transform(model.predict(X_scaler.transform([[
        request.form['length'],
        request.form['diameter'],
        request.form['height'],
        request.form['whole_weight'],
        request.form['shucked_weight'],
        request.form['viscera_weight'],
        request.form['shell_weight'],
        sex_male,
        sex_female,
        sex_infant]])))
    rings = round(rings_a[0][0], 2)
    age = rings + 1.5
    return render_template("index.html", 
        mse = MSE, 
        r2 = r2, 
        length = request.form['length'],
        diameter = request.form['diameter'],
        height = request.form['height'],
        whole_weight = request.form['whole_weight'],
        shucked_weight = request.form['shucked_weight'],
        viscera_weight = request.form['viscera_weight'],
        shell_weight = request.form['shell_weight'],
        sex_male = sex_male_selected,
        sex_female = sex_female_selected,
        sex_infant = sex_infant_selected,
        rings = rings, 
        age = age)

if __name__ == '__main__':
    app.run(debug=True)
