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

RATE_LIMIT_CODES = set([88, 130, 420, 429])

class Twitter:
    def __init__(self, credential_file):
        """
        Params:
          credential_file...list of JSON objects containing the four
          required tokens: consumer_key, consumer_secret, access_token, access_secret
        """
        self.credentials = [json.loads(l) for l in open(credential_file)]
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