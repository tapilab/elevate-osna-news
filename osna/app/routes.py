from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm

# from ..u import get_twitter_data, N_TWEETS
from .. import credentials_path
import sys

from ..mytwitter import Twitter
from .. import credentials_path

# twapi = Twitter(credentials_path)

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()
    result = None
    if form.validate_on_submit():
        input_field = form.input_field.data
        tweets = get_tweets(input_field)
        flash(input_field)
        return render_template('myform.html', title='', form=form, tweets=tweets)
    # return redirect('/index')
    return render_template('myform.html', title='', form=form)


def get_tweets(name='twitterapi'):
    tapi = Twitter(credentials_path)
    tweetsj = tapi._get_tweets('screen_name', name, limit=200)
    tweets = [t['full_text'] for t in tweetsj]
    return tweets

