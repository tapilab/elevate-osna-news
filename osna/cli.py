# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""
import click
import glob
import pickle
import sys
import pandas as pd

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report

from scipy.sparse import csr_matrix, hstack

from osna.clf_train import train_and_predict, make_features, read_data
from osna.get_wordlist import get_text, get_source
from . import credentials_path, clf_path, clf_path2, clf_path3, clf_path_

from keras.layers import Dropout, Flatten
import keras




@click.group()
def main(args=None):
    """Console script for osna."""
    return 0


@main.command('web')
@click.option('-t', '--twitter-credentials', required=False, type=click.Path(exists=True), show_default=True,
              default=credentials_path, help='a json file of twitter tokens')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(twitter_credentials, port):
    from .app import app
    app.run(host='127.0.0.1', debug=True, port=port)


@main.command('stats')
@click.argument('directory', type=click.Path(exists=True))
def stats(directory):
    print('reading from %s' % directory)
    from .stats import Mystats
    Mystats(directory)


@main.command('train')
@click.argument('directory', type=click.Path(exists=True))
def train(directory):
    """
    Train a classifier and save it.
    """
    print('reading from %s' % directory)

    # (1) Read the data...
    df = read_data(directory)

    df = make_features(df)

    text = get_text(list(df.text))
    title = get_text(list(df.title))
    source = get_source(list(df.source))

    # features = df.loc[:, ['avg_retweet', 'avg_favorite','avg_followers','avg_friends','avg_listed']]
    features = df.loc[:, ['avg_retweet', 'avg_favorite', 'var_time', 'var_desc']]
    features = features.to_dict('records')

    # (2) Create classifier and vectorizer.
    # set best parameters
    lr = LogisticRegression(C=10, penalty='l2')
    vec1 = TfidfVectorizer(min_df=2, max_df=.9, ngram_range=(1, 3), stop_words='english')
    vec2 = TfidfVectorizer(min_df=2, max_df=.9, ngram_range=(1, 3), stop_words='english')
    vec3 = CountVectorizer(min_df=1, max_df=.9, ngram_range=(1, 1))
    vecf = DictVectorizer()

    print('fitting...')
    x1 = vec1.fit_transform(text)
    print(x1.shape)
    x2 = vec2.fit_transform(title)
    print(x2.shape)
    x3 = vec3.fit_transform(source)
    print(x3.shape)
    xf = vecf.fit_transform(features)
    print(xf.shape)

    x = hstack([x1, x2, x3, xf])
    x = x.tocsr()
    print(x.shape)

    y = np.array(df.label)

    # (3) do cross-validation and print out validation metrics
    # (classification_report)
    train_and_predict(x, y, lr)

    # (4) Finally, train on ALL data one final time and
    # train...
    clf = train_and_predict(x, y, lr, train=True)

    top_features = []
    features = np.array(vec1.get_feature_names() + vec2.get_feature_names() + vec3.get_feature_names() + vecf.get_feature_names())

    for j in np.argsort(clf.coef_[0][x[0].nonzero()[1]])[::-1][:50]:  # start stop ste
        idx = x[0].nonzero()[1][j]
        top_features.append({'feature': features[idx], 'coef': clf.coef_[0][idx]})
    print(top_features)
    print(pd.DataFrame(top_features))

    # save the classifier
    pickle.dump((vec1, vec2, vec3, vecf, clf), open(clf_path, 'wb'))


@main.command('train_')
@click.argument('directory', type=click.Path(exists=True))
def train(directory):
    """
    Train a classifier and save it.
    """
    print('reading from %s' % directory)

    # (1) Read the data...
    df = read_data(directory)

    df = make_features(df)

    text = get_text(list(df.text))
    title = get_text(list(df.title))
    source = get_source(list(df.source))

    features = df.loc[:, ['avg_retweet', 'avg_favorite', 'var_time', 'var_desc']]
    features = features.to_dict('records')

    # (2) Create classifier and vectorizer.
    # set best parameters
    vec1 = TfidfVectorizer(min_df=2, max_df=.9, ngram_range=(1, 3), stop_words='english')
    vec2 = TfidfVectorizer(min_df=2, max_df=.9, ngram_range=(1, 3), stop_words='english')
    vec3 = CountVectorizer(min_df=1, max_df=.9, ngram_range=(1, 1))
    vecf = DictVectorizer()

    print('fitting...')
    x1 = vec1.fit_transform(text)
    print(x1.shape)
    x2 = vec2.fit_transform(title)
    print(x2.shape)
    x3 = vec3.fit_transform(source)
    print(x3.shape)
    xf = vecf.fit_transform(features)
    print(xf.shape)

    X = hstack([x1, x2, x3, xf])

    np.set_printoptions(threshold=10000)

    X = X.todense()
    X = np.array(X)

    print(X.shape)
    pickle.dump(X, open('X.pkl', 'wb'))

    # X = pickle.load(open('X.pkl', 'rb'))

    y = np.array(df.label)

    for i in range(len(y)):
        if y[i] == 'real':
            y[i] = 0
        else:
            y[i] = 1

    pickle.dump(y, open('y.pkl', 'wb'))

    # y = pickle.load(open('y.pkl', 'rb'))

    # (3) do cross-validation and print out validation metrics
    # (classification_report)

    dropout_rate = .1
    model = keras.Sequential()
    model.add(keras.layers.Dense(16, input_shape=(X.shape[1],)))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    np.random.seed(116)
    np.random.shuffle(X)
    np.random.seed(116)
    np.random.shuffle(y)

    x_val = X[:300]
    partial_x_train = X[300:500]

    y_val = y[:300]
    partial_y_train = y[300:500]

    testX = X[500:]
    testy = y[500:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=158,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)

    results = model.evaluate(testX, testy)

    print(results)

    history_dict = history.history

    import matplotlib.pyplot as plt

    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()  # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # save the classifier
    pickle.dump((vec1, vec2, vec3, vecf, model), open(clf_path_, 'wb'))


@main.command('train2')
@click.argument('directory', type=click.Path(exists=True))
def train(directory):
    df = read_data(directory)

    X = pickle.load(open('X.pkl', 'rb')).tocsr()
    print(X.shape)
    y = df.label
    print(y.shape)

    # (3) do cross-validation and print out validation metrics
    # (classification_report)

    print('MLP----hidden_layer_sizes---')
    accdf = pd.DataFrame(np.random.randn(3, 3), index=['1', '2', '3'],
                         columns=['hidden_layer_sizes', 'Accuracy', 'std'])
    for i, hidden_layer_sizes in zip([0, 1, 2], [10, 50, 100, 200]):
        MP = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,))
        Y = y
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        for train, test in kf.split(X):
            MP.fit(X[train], Y[train])
            pred = MP.predict(X[test])
            accuracies.append(accuracy_score(Y[test], pred))
        mean_acc = np.mean(accuracies)
        std = np.std(accuracies)
        accdf['hidden_layer_sizes'][i] = hidden_layer_sizes
        accdf['Accuracy'][i] = mean_acc
        accdf['std'][i] = std
    print(accdf)

    print('MLP----alpha---')
    accdf = pd.DataFrame(np.random.randn(3, 3), index=['1', '2', '3'],columns=['alpha', 'Accuracy','std'])
    for i,alpha in zip([0,1,2],[.001,.0001,.00001]):
        MP = MLPClassifier(alpha = alpha)
        Y = y
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        for train, test in kf.split(X):
            MP.fit(X[train], Y[train])
            pred = MP.predict(X[test])
            accuracies.append(accuracy_score(Y[test], pred))
        mean_acc = np.mean(accuracies)
        std = np.std(accuracies)
        accdf['alpha'][i] = alpha
        accdf['Accuracy'][i] = mean_acc
        accdf['std'][i] = std
    print(accdf)

    print('RandomForest----min_samples_leaf---')
    accdf = pd.DataFrame(np.random.randn(3, 3), index=['1', '2', '3'], columns=['min_samples_leaf', 'Accuracy', 'std'])

    for i, min_samples_leaf in zip([0, 1, 2], [1, 3, 5]):
        RFC = RandomForestClassifier(min_samples_leaf=min_samples_leaf)

        Y = y
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        for train, test in kf.split(X):
            RFC.fit(X[train], Y[train])
            pred = RFC.predict(X[test])
            accuracies.append(accuracy_score(Y[test], pred))
        mean_acc = np.mean(accuracies)
        std = np.std(accuracies)
        accdf['min_samples_leaf'][i] = min_samples_leaf
        accdf['Accuracy'][i] = mean_acc
        accdf['std'][i] = std
    print(accdf)


    print('RandomForest----n_estimators---')
    accdf = pd.DataFrame(np.random.randn(3, 3), index=['1', '2', '3'], columns=['n_estimators', 'Accuracy', 'std'])
    for i, n_estimators in zip([0, 1, 2], [100, 200, 300]):
        #     print('==================n_estimators : %d ================' %(n_estimators))
        RFC = RandomForestClassifier(n_estimators=n_estimators)

        Y = y
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        for train, test in kf.split(X):
            RFC.fit(X[train], Y[train])
            pred = RFC.predict(X[test])
            accuracies.append(accuracy_score(Y[test], pred))
        #         print(classification_report(Y[test], pred))
        #     print('accuracy over all cross-validation folds: %s' % str(accuracies))
        mean_acc = np.mean(accuracies)
        std = np.std(accuracies)
        #     print('mean=%.2f std=%.2f' % (mean_acc, std))
        accdf['n_estimators'][i] = n_estimators
        accdf['Accuracy'][i] = mean_acc
        accdf['std'][i] = std
    print(accdf)





# @main.command('train2')
# @click.argument('directory', type=click.Path(exists=True))
# def train(directory):
#     """
#     Train a classifier and save it.
#     """
#     print('reading from %s' % directory)
#
#     # (1) Read the data...
#     df = read_data(directory)
#
#     # (2) Create classifier and vectorizer.
#     # set best parameters
#     lr = LogisticRegression(C=10, penalty='l2')
#     vecf = DictVectorizer()
#
#     df = make_features(df)
#
#     features = df.loc[:, ['avg_retweet', 'avg_favorite', 'var_time', 'var_desc']]
#     features = features.to_dict('records')
#     # print(features)
#
#     # now use some user feature
#     x = vecf.fit_transform(features)
#     print(x.shape)
#
#     y = np.array(df.label)
#
#     # (3) do cross-validation and print out validation metrics
#     # (classification_report)
#     train_and_predict(x, y, lr)
#
#     # (4) Finally, train on ALL data one final time and
#     # train...
#     clf = train_and_predict(x, y, lr, train=True)
#     # save the classifier
#     pickle.dump((vecf, clf), open(clf_path2, 'wb'))


# @main.command('train3')
# @click.argument('directory', type=click.Path(exists=True))
# def train(directory):
#     """
#     Train a classifier and save it.
#     """
#     print('reading from %s' % directory)
#
#     # (1) Read the data...
#     df = read_data(directory)
#
#     text = get_text(list(df.text))
#     title = get_text(list(df.title))
#     source = get_source(list(df.source))
#
#     # (2) Create classifier and vectorizer.
#     # set best parameters
#     lr = LogisticRegression(C=10, penalty='l2')
#     vec1 = TfidfVectorizer(min_df=2, max_df=.9, ngram_range=(1, 3), stop_words='english')
#     vec2 = TfidfVectorizer(min_df=2, max_df=.9, ngram_range=(1, 3), stop_words='english')
#     vec3 = CountVectorizer(min_df=1, max_df=.9, ngram_range=(1, 1))
#
#     x1 = vec1.fit_transform(text)
#     # print(x1.shape)
#     x2 = vec2.fit_transform(title)
#     # print(x2.shape)
#     x3 = vec3.fit_transform(source)
#     # print(x3.shape)
#
#     # features = make_features(df)
#
#     # x = np.hstack([x1, x2, x3, features])
#     # x = hstack([x1, x2, x3])
#
#     ## now only use news texts
#     x = x1
#     print(x.shape)
#     y = np.array(df.label)
#
#     # (3) do cross-validation and print out validation metrics
#     # (classification_report)
#     train_and_predict(x, y, lr)
#
#     # (4) Finally, train on ALL data one final time and
#     # train...
#     clf = train_and_predict(x, y, lr, train=True)
#     # save the classifier
#     pickle.dump((vec1, vec2, vec3, clf), open(clf_path3, 'wb'))


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
