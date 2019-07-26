from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
from .. import credentials_path, clf_path

import pickle
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
    result = None

    if form.validate_on_submit():

        input_field = form.input_field.data
        flash(input_field)
        t = Twitter(credentials_path)
        print(credentials_path)
        #t = Twitter('/Users/mike/elevate-osna-news/credentials.json')
        tweets = [tweet['full_text'] for tweet in t._get_tweets('screen_name', input_field, limit=200)]
        return render_template('myform.html', title='', form=form, tweets=tweets)
    # return redirect('/index')

    return render_template('myform.html', title='', form=form)
