
# importing libraries
import os
import numpy as np
import flask
import pickle
import pandas as pd
from flask import Flask, render_template, request

# creating instance of the class
app = Flask(__name__)

# to tell flask what url shoud trigger the function index()

attr_list = pickle.load(open("./attr_list.pkl", "rb"))
city_list = pickle.load(open("./city_list.pkl", "rb"))


@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


def transform_data(city, attr):
    x = [0] * 262
    x[200+city_list.index(city)] = 1
    attrs = [a.strip() for a in attr.split(',')]
    for a in attrs:
        x[attr_list.index(a)] = 1
    return x


def ValuePredictor(input):

    # df = pd.DataFrame([to_predict_list], columns=['City', 'Category'])

    stars_model = pickle.load(open("./stars.pkl", "rb"))
    comments_model = pickle.load(open("./comments.pkl", "rb"))

    stars_result = stars_model.predict([input])
    comments_result = comments_model.predict([input])

    print(stars_result, comments_result)

    return stars_result, comments_result


@app.route('/result', methods=['POST'])
def result():
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    # print(to_predict_list)it
    input1 = to_predict_list[0]
    input2 = to_predict_list[1]
    input = transform_data(input1, input2)
    # category = transform_data(input2)

    # print(city, category)

    stars_result, comments_result = ValuePredictor(input)

    # table = a.to_html(classes='data', header='true')
    return render_template("result.html", city=to_predict_list[0], type=to_predict_list[1], review=comments_result, stars=stars_result)


if __name__ == '__main__':
    app.run()
