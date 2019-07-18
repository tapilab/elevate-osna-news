import pandas as pd
import re
from collections import Counter
from tqdm import tqdm

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

    # load check data
    ck = pd.read_csv('..\\training_data\\factchecks.csv')
    # print(ck)
    # print(ck.describe())
    # print(ck.keys())
    # print(df['comment_tokens'])

    ck = ck.loc[ck['site'] == df['site'][0], ['site', 'social_id', 'ruling']]
    ck['social_id'] = ck['social_id'].astype(df['social_id'].dtype)

    ck = pd.merge(ck, df, on=['social_id', 'site'], how='outer')
    ck.fillna('unknown', inplace=True)
    # print(ck.keys())
    # print(ck)

    ct = ck.loc[((ck['ruling'] == 'True') | (ck['ruling'] == 'TRUE') | (ck['ruling'] == 'true')) & (
            ck['comment_tokens'] != 'unknown')]
    # print('true sets\n', ct)
    cf = ck.loc[((ck['ruling'] == 'FALSE') | (ck['ruling'] == 'False') | (ck['ruling'] == 'false')) & (
            ck['comment_tokens'] != 'unknown')]
    # print('false sets\n', cf)
    cu = ck.loc[ck['ruling'] == 'unknown']
    # print('unknown sets\n', cu)

    counts = counters(df['social_id'])
    print('Number of unique users:', len(counts))
    counts = counters(df['comment_tokens'])
    print('Number of unique messages:', len(counts))

    counts = counters(ct['social_id'])
    print('\nNumber of unique users in TRUE:', len(counts))
    counts = counters(cf['social_id'])
    print('Number of unique users in FALSE:', len(counts))
    counts = counters(cu['social_id'])
    print('Number of unique users in unknown:', len(counts))

    counts = counters(ct['comment_tokens'])
    print('\nNumber of unique messages in TRUE:', len(counts))
    counts = counters(cf['comment_tokens'])
    print('Number of unique messages in FALSE:', len(counts))
    counts = counters(cu['comment_tokens'])
    print('Number of unique messages in unknown:', len(counts),'\n')

    tokens = [token for tweet in tqdm(df['comment_tokens'], ncols=80) for token in tweet_tokenizer(tweet)]
    counts = counters(tokens)

    print('Number of unique words:', len(counts))
    print('Number of unique tokens:', len(tokens))
    print('\n50 most common words:', counts.most_common(50))

    tokens = [token for tweet in tqdm(ct['comment_tokens'], ncols=80) for token in tweet_tokenizer(tweet)]
    counts = counters(tokens)

    print('50 most common words in TRUE:', counts.most_common(50))

    tokens = [token for tweet in tqdm(cf['comment_tokens'], ncols=80) for token in tweet_tokenizer(tweet)]
    counts = counters(tokens)
    print('50 most common words in FALSE:', counts.most_common(50))

    tokens = [token for tweet in tqdm(cu['comment_tokens'], ncols=80) for token in tweet_tokenizer(tweet)]
    counts = counters(tokens)
    print('50 most common words in unknown:', counts.most_common(50))

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


# if __name__ == '__main__':
    # panda's display settings
    # pd.set_option('display.max_rows', 10)
    # pd.set_option('display.max_columns', 10)

    # unit test
    # DATAPATH = '..\\..\\training_data\\twitter.csv'
    # df = Mystats(DATAPATH)
