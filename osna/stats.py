import pandas as pd
import re
from collections import Counter

# read json
# import json
# f = open(directory, 'r', encoding='utf-8')
# tweets = []
# for line in f:
#     tweets.append(json.load(line))
# print(tweets)

def Mystats(directory):
    df = pd.read_csv(directory)
    # print(df)
    # print(df.describe())
    # print(df.keys())
    # print(df['comment_tokens'])

    counts = counters(df['social_id'])
    print('Number of unique users:', len(counts))
    counts = counters(df['comment_tokens'])
    print('Number of unique messages:', len(counts))

    tokens = []
    for tweet in df['comment_tokens']:
        tokens = tokens + tweet_tokenizer(tweet)
        # print(len(tokens))
        ## for test purpose
        if len(tokens) > 10000:
            break
    counts = counters(tokens)
    print('Number of unique words:', len(counts))
    print('Number of unique tokens:', len(tokens))
    print('50 most common words:', counts.most_common(50))

    return 0


def counters(d):
    counts = Counter()  # handy object: dict from object -> int
    counts.update(d)
    return counts


def tweet_tokenizer(s):
    s = re.sub(r'#(\S+)', r'HASHTAG_\1', s)
    s = re.sub(r'@(\S+)', r'MENTION_\1', s)
    s = re.sub(r'http\S+', 'THIS_IS_A_URL', s)
    return re.sub('\W+', ' ', s.lower()).split()


if __name__ == '__main__':
    DATAPATH = 'D:\\python\\training_data\\twitter.csv'
    df = Mystats(DATAPATH)
    # counters(df, 'social_id')
