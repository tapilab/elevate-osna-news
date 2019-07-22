from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
# from ..mytwitter import Twitter
# from ..u import get_twitter_data, N_TWEETS
from .. import credentials_path
import sys
import json
from TwitterAPI import TwitterAPI
from ..mytwitter import Twitter


# twapi = Twitter(credentials_path)

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()
    result = None

    if form.validate_on_submit():
        chk, tweets = getTwt(form.input_field.data)
        if chk == 0:
            input_field = tweets
            flash(input_field)
            return render_template('myform.html', title='', form=form)
        else:
            info = str(len(tweets)) + " tweet(s) detected for " + form.input_field.data + " in total :"
            flash(info)
            for each in tweets:
                input_field = each
                flash(input_field)
    return render_template('myform.html', title='', form=form)


def getTwt(user_name):
    """
    Establing connection
    """
    path = "credentials.json"
    file = json.loads(open(path).read())
    twt = TwitterAPI(file["consumer_key"], file["consumer_secret"], file["access_token"], file["token_secret"])

    """
    Acquiring user's info 
    """
    max_id = None
    param = {'screen_name': user_name, 'max_id': max_id, 'tweet_mode': 'extended', 'trim_user': 0, 'count': 50}
    ret = []
    lst_len = 0
    while len(ret) < 200:
        req = twt.request('statuses/user_timeline', param)
        if req.status_code == 200:
            for each in req:
                ret.append(each['full_text'])
        else:
            return 0, "NO SUCH USER!"

        if len(ret) - lst_len > 0:
            print(len(ret) - lst_len)
            param['max_id'] = min(t['id'] for t in req) - 1
            if len(ret) - lst_len < 48:
                if len(ret) < 200:
                    return 1, ret
                else:
                    return 1, ret[0:200]
            lst_len = len(ret)

        else:
            break
    # return 1,ret
    if len(ret) < 200:
        return 1, ret
    else:
        return 1, ret[0:200]
