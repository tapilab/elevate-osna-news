from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
# from ..mytwitter import Twitter
# from ..u import get_twitter_data, N_TWEETS
from .. import credentials_path
import sys


# twapi = Twitter(credentials_path)

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()
    result = None
    if form.validate_on_submit():
        input_field = form.input_field.data
        flash(input_field)
        return render_template('myform.html', title='', form=form)
    # return redirect('/index')
    return render_template('myform.html', title='', form=form)

