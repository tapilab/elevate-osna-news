from flask import render_template, flash, redirect, session

from osna.clf_train import make_features
from . import app
from .forms import MyForm
from .. import credentials_path, clf_path, clf_path2
from osna.get_wordlist import get_text,get_source

import pickle
import numpy as np
import pandas as pd
import sys
import json
from TwitterAPI import TwitterAPI
from ..mytwitter import Twitter

# clf, vec = pickle.load(open(clf_path, 'rb'))
# print('read clf %s' % str(clf))
# print('read vec %s' % str(vec))

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()

    if form.validate_on_submit():

        input_field = form.input_field.data
        method = form.select_field.data
        flash(input_field)

        # tweets = [tweet['full_text'] for tweet in t._get_tweets('screen_name', input_field, limit=200)]

        news = get_tweets(input_field)
        pred, proba, top_features = predict(news)

        return render_template('myform.html', title='', form=form, news=news, pred=pred, proba=max(proba*100), top_features=top_features)
    # return redirect('/index')

    return render_template('myform.html', title='', form=form)


def get_tweets(input_field):
    t = Twitter(credentials_path)
    #search news and get tweets
    new_tweets = t._search_news(input_field)
    return new_tweets


def predict(df):
    vec1, vec2, vec3, lr = pickle.load(open(clf_path, 'rb'))

    text = get_text(list(df.text))
    # title = get_text(list(df.title))
    # source = get_source(list(df.source))

    x1 = vec1.transform(text)
    # x2 = vec2.transform(title)
    # x3 = vec3.transform(source)

    # features = make_features(df)

    # x = np.hstack([x1, x2, x3, features])
    x = x1

    top_features = []

    pred = lr.predict(x)[0]
    proba = lr.predict_proba(x)[0]

    top_features = []
    features = vec1.get_feature_names()

    for j in np.argsort(lr.coef_[0][x[0].nonzero()[1]])[::-1][:3]:  # start stop step
        idx = x[0].nonzero()[1][j]
        top_features.append({'feature': features[idx], 'coef': lr.coef_[0][idx]})

    return pred, proba, top_features

def predict1(df):
    vecf, lr = pickle.load(open(clf_path2, 'rb'))

    features = make_features(df)

    x1 = vecf.transform(features)
    # x2 = vec2.transform(title)
    # x3 = vec3.transform(source)

    # features = make_features(df)

    # x = np.hstack([x1, x2, x3, features])
    x = x1

    top_features = []

    pred = lr.predict(x)[0]
    proba = lr.predict_proba(x)[0]

    top_features = []
    features = vec1.get_feature_names()

    for j in np.argsort(lr.coef_[0][x[0].nonzero()[1]])[::-1][:3]:  # start stop step
        idx = x[0].nonzero()[1][j]
        top_features.append({'feature': features[idx], 'coef': lr.coef_[0][idx]})

    return pred, proba, top_features





