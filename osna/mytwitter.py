"""
Wrapper for Twitter API.
"""
from itertools import cycle
import json
import requests
import sys
import time
import traceback
from TwitterAPI import TwitterAPI
import pandas as pd
from goose3 import Goose

from osna import num_limit

RATE_LIMIT_CODES = set([88, 130, 420, 429])


class Twitter:
    def __init__(self, credential_file):
        """
        Params:
          credential_file...list of JSON objects containing the four
          required tokens: consumer_key, consumer_secret, access_token, access_secret
        """
        self.credentials = [json.loads(l) for l in open(credential_file)]
        print(self.credentials)
        self.credential_cycler = cycle(self.credentials)
        self.reinit_api()

    def reinit_api(self):
        creds = next(self.credential_cycler)
        sys.stderr.write('switching creds to %s\n' % creds['consumer_key'])
        self.twapi = TwitterAPI(creds['consumer_key'],
                                creds['consumer_secret'],
                                creds['access_token'],
                                creds['token_secret'])

    def request(self, endpoint, params):
        while True:
            try:
                response = self.twapi.request(endpoint, params)
                if response.status_code in RATE_LIMIT_CODES:
                    for _ in range(len(self.credentials) - 1):
                        self.reinit_api()
                        response = self.twapi.request(endpoint, params)
                        if response.status_code not in RATE_LIMIT_CODES:
                            return response
                    sys.stderr.write('sleeping for 15 minutes...\n')
                    time.sleep(910)  # sleep for 15 minutes # FIXME: read the required wait time.
                    return self.request(endpoint, params)
                else:
                    return response
            except requests.exceptions.Timeout:
                # handle requests.exceptions.ConnectionError Read timed out.
                print("Timeout occurred. Retrying...")
                time.sleep(5)
                self.reinit_api()

    def followers_for_id(self, theid, limit=1e10):
        return self._get_followers('user_id', theid, limit)

    def followers_for_screen_name(self, screen_name, limit=1e10):
        return self._get_followers('screen_name', screen_name, limit)

    def _get_followers(self, identifier_field, identifier, limit=1e10):
        return self._paged_request('followers/ids',
                                   {identifier_field: identifier,
                                    'count': 5000,
                                    'stringify_ids': True},
                                   limit)

    def friends_for_id(self, theid, limit=1e10):
        return self._get_friends('user_id', theid, limit)

    def friends_for_screen_name(self, screen_name, limit=1e10):
        return self._get_friends('screen_name', screen_name, limit)

    def _get_friends(self, identifier_field, identifier, limit=1e10):
        return self._paged_request('friends/ids',
                                   {identifier_field: identifier,
                                    'count': 5000,
                                    'stringify_ids': True},
                                   limit)

    def _paged_request(self, endpoint, params, limit):
        results = []
        cursor = -1
        while len(results) <= limit:
            try:
                response = self.request(endpoint, params)
                if response.status_code != 200:
                    sys.stderr.write('Skipping bad request: %s\n' % response.text)
                    return results
                else:
                    result = json.loads(response.text)
                    items = [r for r in response]
                    if len(items) == 0:
                        return results
                    else:
                        sys.stderr.write('fetched %d more results for %s\n' %
                                         (len(items),
                                          params['screen_name'] if 'screen_name' in params else params['user_id']))
                        time.sleep(1)
                        results.extend(items)
                    params['cursor'] = result['next_cursor']
            except Exception as e:
                sys.stderr.write('Error: %s\nskipping...\n' % e)
                sys.stderr.write(traceback.format_exc())
                return results
        return results

    def tweets_for_id(self, theid, limit=1e10):
        return self._get_tweets('user_id', theid, limit)

    def tweets_for_screen_name(self, screen_name, limit=1e10):
        return self._get_tweets('screen_name', screen_name, limit)

    def _get_tweets(self, identifier_field, identifier, limit=1e10):
        max_id = None
        tweets = []
        while len(tweets) < limit:
            try:
                params = {identifier_field: identifier, 'count': 200,
                          'max_id': max_id, 'tweet_mode': 'extended', 'trim_user': 0}
                if max_id:
                    params['max_id'] = max_id
                response = self.request('statuses/user_timeline', params)
                if response.status_code == 200:  # success
                    items = [t for t in response]
                    if len(items) > 0:
                        sys.stderr.write('fetched %d more tweets for %s\n' % (len(items), identifier))
                        tweets.extend(items)
                    else:
                        return tweets
                    max_id = min(t['id'] for t in response) - 1
                else:
                    sys.stderr.write('Skipping bad user: %s\n' % response.text)
                    return tweets
            except Exception as e:
                sys.stderr.write('Error: %s\nskipping...\n' % e)
                sys.stderr.write(traceback.format_exc() + '\n')
                return tweets
        return tweets

    def _search_news(self, identifier):
        tweets = []
        since_id = 0
        tweets_num = 0
        while tweets_num < num_limit:
            response = self.request('search/tweets',{'q':identifier,'count':100,'since_id':since_id})
            if response.status_code == 200:  # success
                items = [t for t in response]
                if len(items) == 100:
                    tweets.extend(items)
                    since_id = tweets[-1]['user']['id'] + 1
                    tweets_num = len(tweets)
                else:
                    tweets.extend(items)
                    break
            else:
                sys.stderr.write('error')
                break

        g = Goose()
        article = g.extract(url=identifier)
        title = article.title
        text = article.cleaned_text

        df = pd.DataFrame({'source': identifier, 'title': title, 'text': text, 'tweets': [tweets]})

        return df

    def _search_tweets(self, identifier):
        since_id = 0
        response = self.request('search/tweets', {'q': identifier, 'count': 15, 'since_id': since_id})
        if response.status_code == 200:  # success
            return [t for t in response][0]



