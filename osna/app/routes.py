from flask import render_template, flash, redirect, session

from osna.clf_train import make_features
from . import app
from .forms import MyForm
from .. import credentials_path, clf_path

import pickle
import numpy as np
import pandas as pd
import sys
import json
from TwitterAPI import TwitterAPI
from ..mytwitter import Twitter

clf, vec = pickle.load(open(clf_path, 'rb'))
print('read clf %s' % str(clf))
print('read vec %s' % str(vec))

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()

    if form.validate_on_submit():

        input_field = form.input_field.data
        flash(input_field)

        # tweets = [tweet['full_text'] for tweet in t._get_tweets('screen_name', input_field, limit=200)]

        news = get_tweets(input_field)

        pred, proba = predict(pd.DataFrame(news))

        return render_template('myform.html', title='', form=form, news=news, pred=pred, proba=proba*100)
    # return redirect('/index')

    return render_template('myform.html', title='', form=form)


def get_tweets(input_field):
    t = Twitter(credentials_path)
    new_tweets = t._get_tweets('screen_name', input_field, limit=200)
    return new_tweets


def predict(df):
    vec1, vec2, vec3, lr = pickle.load(open('clf.pkl', 'rb'))

    x1 = vec1.transform(df.text)
    x2 = vec2.transform(df.source)
    x3 = vec3.transform(df.title)

    features = make_features(df)

    x = np.hstack([x1, x2, x3, features])

    pred = lr.predict(x)[0]
    proba = lr.predict_proba(x)[0]

    return pred, proba





