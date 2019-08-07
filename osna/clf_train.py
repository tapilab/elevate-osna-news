import glob
import os
import json
import gzip
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from scipy.sparse import csr_matrix, hstack
import pickle

from osna.get_wordlist import get_desc
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import time


def load_data(datafile, checkfile):
    """
    Read your data into a single pandas dataframe where
    - each row is an instance to be classified

    (this could be a tweet, user, or news article, depending on your project)
    - there is a column called `label` which stores the class label (e.g., the true
      category for this row)
    """
    df = pd.read_csv(datafile)[['social_id', 'comment_tokens', 'comment_time']]
    ck = pd.read_csv(checkfile)

    ck = ck.loc[ck['site'] == 'twitter', ['site', 'social_id', 'ruling_val']]

    ck['social_id'] = ck['social_id'].astype(df['social_id'].dtype)

    df.columns = ['id', 'text', 'time']
    #     df = df.drop_duplicates(['id','text'])
    ck.columns = ['site', 'id', 'label']
    df = pd.merge(ck, df, on=['id'], how='inner')
    df['label'] = ['true' if i > 0 else 'false' if i < 0 else 'unknown' for i in df.label]
    df['comments_count'] = 1
    df['timemin'] = df['time']

    # combine multiple rows of an id into one row
    def ab(df):
        return ' '.join(df.values)

    df = df.groupby(['id', 'label']).agg({'text': ab, 'comments_count': sum, 'time': max, 'timemin': min})

    def normalization(x, Max, Min):
        s = np.round((x - Min) / (Max - Min), 2)
        return s

    Max = max(df.comments_count)
    Min = min(df.comments_count)
    df['comments_count'] = [normalization(i, Max, Min) for i in df.comments_count]

    Max = max(df.comments_count)
    Min = min(df.comments_count)
    df['timeslot'] = df['time'] - df['timemin']
    df['timeslot'] = [normalization(i, Max, Min) for i in df.timeslot]

    # df['timepercomm'] = df['timeslot'] / df['comments_count']
    # df['timepercomm'] = [int(i) for i in df.timepercomm]

    df = df.drop(['time', 'timemin'], axis=1)
    df = df.reset_index()

    return df


def read_data(directory):
    dfs = []
    for label in ['real', 'fake']:
        for file in glob.glob(directory + os.path.sep + label + os.path.sep + '*gz'):
            print('reading %s' % file)
            df = pd.DataFrame((json.loads(line) for line in gzip.open(file)))
            df['label'] = label
            dfs.append(df)
    df = pd.concat(dfs)[['publish_date', 'source', 'text', 'title', 'tweets', 'label']]
    list_text = [i for i in list(df.text) if i != '']
    return df[df.text.isin(list_text)]


# # load data
# df = load_data('..\\..\\training_data\\twitter.csv', '..\\..\\training_data\\factchecks.csv')
# # print(df.head(), df.keys())


def train_and_predict(X, Y, lr, train=False):
    # vectorize text
    # vec = TfidfVectorizer(analyzer='word', token_pattern=r'[^0-9_\W]+', min_df=1)
    # X = vec.fit_transform(df.text)

    # add features
    # features = np.matrix([df.timeslot, df.comments_count]).T
    # print(features)
    # X = hstack([X, features])
    X = X.todense()
    # X = features
    # load lables
    # Y = np.array(df.label)

    # fit our classifier.

    # lr.fit(X, Y)

    # # this will be saved to disc and loaded by our web app.
    # pickle.dump((vec, lr), open('clf.pkl', 'wb'))
    #
    # # now, when we run the web app, we first load the vec and clf
    # # from the file.
    # vec, lr = pickle.load(open('clf.pkl', 'rb'))

    if not train:
        from sklearn.model_selection import KFold
        from sklearn.metrics import accuracy_score

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        report = []
        for train, test in kf.split(X):
            lr.fit(X[train], Y[train])
            pred = lr.predict(X[test])
            accuracies.append(accuracy_score(Y[test], pred))
            report.append(classification_report(Y[test], pred))
        for r in report:
            print(r)
        print('accuracy over all cross-validation folds: %s' % str(accuracies))
        print('mean=%.2f std=%.2f' % (np.mean(accuracies), np.std(accuracies)))
        return np.mean(accuracies)
    elif train:
        lr.fit(X, Y)
        return lr
    else:
        ex = Exception('wrong command')
        raise ex


def make_features(df):
    ## Add your code to create features.
    features: np.matrix
    avg_ret = []
    avg_fav = []
    var_desc = []
    var_time = []
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 1))
    print('Extracting features...')
    for j in tqdm(range(len(df)), ncols=80):
        tweets = df.tweets.values[j]
        retweet = []
        favorite = []
        time=[]
        ##timing = []
        list_desc = []
        if len(tweets) > 1:
            for i in range(len(tweets)):
                retweet.append(tweets[i]['retweet_count'])
                favorite.append(tweets[i]['favorite_count'])
                ##a=tweets[i]['created_at'][-4:]+tweets[i]['created_at'][4:7]+tweets[i]['created_at'][8:10]+tweets[i]['created_at'][11:13]
                ##timing.append(time.strftime("%Y%m%d%H",time.strptime(a,"%Y%b%d%H")))
                time.append(tweets[i]['created_at'][4:19] + tweets[i]['created_at'][-5:])
                if 'description' in list(tweets[i]['user'].keys()):
                    description = get_desc(tweets[i]['user']['description'])
                    list_desc.append(description)
            ##timing=sorted(timing)
            ##timing=[timing[i] for i in range(len(timing)) if int(timing[i])<= int(timing[0])+200]
            avg_ret.append(sum(retweet) / len(tweets))
            avg_fav.append(sum(favorite) / len(tweets))
            time_sums = [v for k, v in Counter(time).items()]
            ##time_sums = [v for k, v in Counter(timing).items()]
            var_time.append(np.var(time_sums))
            if len(list_desc) > 1:
                X = vec.fit_transform(list_desc)
                sim = cosine_similarity(X)
                var_desc.append(np.var(sim))
            else:
                var_desc.append(0.0)
        elif len(tweets) == 1:
            avg_ret.append(sum(retweet))
            avg_fav.append(sum(favorite))
            var_time.append(0.0)
            var_desc.append(0.0)
        else:
            avg_ret.append(0.0)
            avg_fav.append(0.0)
            var_time.append(0.0)
            var_desc.append(0.0)

    df['avg_retweet'] = avg_ret
    df['avg_favorite'] = avg_fav
    df['var_time'] = var_time
    df['var_desc'] = var_desc

    return df


def quantization(f, bins):
    features = []
    for key in f.keys():
        cats = pd.cut(f[key], bins, labels=False)
        for i in range(len(bins)):
            features.append([1 if c == i else 0 for c in cats])
    features = np.matrix(features).T
    return features
#
# df = read_data('..\\..\\training_data_2')
# text = df.text
# y = np.array(df.label)
#
# print('----min_df---')
# accdf1 = pd.DataFrame(index=['1', '2', '3','4'], columns=['index','min_df', 'Accuracy'])
# for i, min_df in enumerate([1, 2, 5, 10]):
#     print(min_df,'\t', end='')
#     lr = LogisticRegression(C=1, penalty='l2')
#     vec = TfidfVectorizer(analyzer='word',token_pattern=r'[^0-9_\W]+', min_df=min_df, max_df=1., ngram_range=(1, 1))
#     x = vec.fit_transform(text)
#     acc = train_and_predict(x, y, lr)
#     accdf1['index'][i] = i
#     accdf1['min_df'][i] = min_df
#     accdf1['Accuracy'][i] = acc
# print(accdf1)
# ax = accdf1.plot(x = 'index', y='Accuracy', legend='min_df')
#
# print('----max_df---')
# accdf2 = pd.DataFrame(index=['1', '2', '3'], columns=['index','max_df', 'Accuracy'])
# for i, max_df in enumerate([1., .9, .8]):
#     print(max_df, '\t', end='')
#     lr = LogisticRegression(C=1, penalty='l2')
#     vec = TfidfVectorizer(analyzer='word',token_pattern=r'[^0-9_\W]+',min_df=2, max_df=max_df, ngram_range=(1, 1))
#     x = vec.fit_transform(text)
#     acc = train_and_predict(x, y, lr)
#     accdf2['index'][i] = i
#     accdf2['max_df'][i] = max_df
#     accdf2['Accuracy'][i] = acc
# print(accdf2)
# accdf2.plot(x = 'index', y='Accuracy',legend='max_df', ax=ax)
#
# print('----ngram_range---')
# accdf3 = pd.DataFrame(index=['1', '2', '3'], columns=['index','ngram_range', 'Accuracy'])
# for i, ngram_range in enumerate([(1,1), (1,2), (1,3)]):
#     print(ngram_range, '\t', end='')
#     lr = LogisticRegression(C=1, penalty='l2')
#     vec = TfidfVectorizer(analyzer='word',token_pattern=r'[^0-9_\W]+',min_df=2, max_df=1., ngram_range=ngram_range)
#     x = vec.fit_transform(text)
#     acc = train_and_predict(x, y, lr)
#     accdf3['index'][i] = i
#     accdf3['ngram_range'][i] = ngram_range
#     accdf3['Accuracy'][i] = acc
# print(accdf3)
# # accdf3.plot(x = 'index', y='Accuracy',legend='ngram_range', ax=ax)
#
# print('----C---')
# accdf4 = pd.DataFrame(index=['1', '2', '3','4'], columns=['index','C', 'Accuracy'])
# for i, C in enumerate([.1, 1, 5, 10]):
#     print(C, '\t', end='')
#     lr = LogisticRegression(C=C, penalty='l2')
#     vec = TfidfVectorizer(analyzer='word',token_pattern=r'[^0-9_\W]+', min_df=2, max_df=1., ngram_range=(1, 1))
#     x = vec.fit_transform(text)
#     acc = train_and_predict(x, y, lr)
#     accdf4['index'][i] = i
#     accdf4['C'][i] = C
#     accdf4['Accuracy'][i] = acc
# print(accdf4)
# accdf4.plot(x = 'index', y='Accuracy',legend='C', ax=ax)
#
# print('----penalty---')
# accdf5 = pd.DataFrame(index=['1', '2'], columns=['index','penalty', 'Accuracy'])
# for i, penalty in enumerate(['l1', 'l2']):
#     print(penalty, '\t', end='')
#     lr = LogisticRegression(C=1, penalty=penalty)
#     vec = TfidfVectorizer(analyzer='word',token_pattern=r'[^0-9_\W]+', min_df=2, max_df=1., ngram_range=(1, 1))
#     x = vec.fit_transform(text)
#     acc = train_and_predict(x, y, lr)
#     accdf5['index'][i] = i
#     accdf5['penalty'][i] = penalty
#     accdf5['Accuracy'][i] = acc
# print(accdf5)
# # accdf5.plot(x = 'index', y='Accuracy',legend='penalty', ax=ax)
#
# plt.legend()
# plt.show()


# lr = LogisticRegression(C=10, penalty='l2')
# vec = TfidfVectorizer(analyzer='word', token_pattern=r'[^0-9_\W]+', min_df=2, max_df=.9, ngram_range=(1, 3))
# train_and_predict(df, vec, lr)

## optimized classifier
# lr = LogisticRegression(C=10, penalty='l2')
# vec = TfidfVectorizer(analyzer='word', token_pattern=r'[^0-9_\W]+', min_df=2, max_df=1., ngram_range=(1, 3))
# train_and_predict(vec, lr)

# it is important to call `transform`, and NOT `fit_transform`
# here. fit_transform will learn a new vocabulary,
# which will change the columns of the feature
# matrix that the classifier expects.


# new_tweet = vec.transform([{'f1': 7, 'f2': -1, 'f3': 10}])
# print('new tweet feature vector:', new_tweet.todense())
# prediction = lr.predict(new_tweet)[0]
# print('prediction=', prediction)
# probas = lr.predict_proba(new_tweet)
# print('probas=', probas)
# print('predicted %s witih probability %.2f' % (prediction, probas.max()))


# notice that the 'f3' feature was not present at training
# time, so it is ignored for the new example.
