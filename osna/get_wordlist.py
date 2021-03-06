import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

def tokennizer(s):
    s = re.sub(r'http\S+', '', s)
    s = re.sub(r'[0-9_\s]+', '', s)
    s = re.sub(r"[^'\w]+", '', s)

    s = re.compile(r"(?<=[a-zA-Z])'re").sub(' are', s)
    s = re.compile(r"(?<=[a-zA-Z])'m").sub(' am', s)
    s = re.compile(r"(?<=[a-zA-Z])'ve").sub(' have', s)
    s = re.compile(r"(it|he|she|that|this|there|here|what|where|when|who|why|which)('s)").sub(r"\1 is", s)
    s = re.sub(r"'s", "", s)
    s = re.sub(r"can't", 'can not', s)
    s = re.compile(r"(?<=[a-zA-Z])n't").sub(' not', s)
    s = re.compile(r"(?<=[a-zA-Z])'ll").sub(' will', s)
    s = re.compile(r"(?<=[a-zA-Z])'d").sub(' would', s)
    return s


def lemmatize(l):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(l):
        if tag.startswith('NN'):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            yield wnl.lemmatize(word, pos='r')
        else:
            yield word


def tokennizer_desc(tknzr,sen):
    list1=[]
    for s in tknzr.tokenize(sen.lower()):
        s=re.compile(r"(?<=[a-zA-Z])'re").sub(' are',s)
        s=re.compile(r"(?<=[a-zA-Z])'m").sub(' am',s)
        s=re.compile(r"(?<=[a-zA-Z])'ve").sub(' have',s)
        s=re.compile(r"(it|he|she|that|this|there|here|what|where|when|who|why|which)('s)").sub(r"\1 is",s)
        s=re.sub(r"'s","",s)
        s=re.sub(r"can't",'can not',s)
        s=re.compile(r"(?<=[a-zA-Z])n't").sub(' not',s)
        s=re.compile(r"(?<=[a-zA-Z])'ll").sub(' will',s)
        s=re.compile(r"(?<=[a-zA-Z])'d").sub(' would',s)
        list1.append(s)
    return(' '.join(list1))


def get_text(list):
    stopword=set(stopwords.words('english'))
    list_new=[]
    for l in list:
        l=re.sub(r"[^\w']",' ',l).lower()
        l1=[tokennizer(w) for w in l.split() if len(tokennizer(w))>2]
        l=' '.join(l1)
        l1=[tokennizer(w) for w in l.split() if len(tokennizer(w))>2 and tokennizer(w) not in stopword]
        l=' '.join(lemmatize(l1))
        list_new.append(l)
    return list_new

def get_source(list):
    list_new = []
    for l in list:
        l = re.sub(r"http[s]?://", '', l).lower()
        l1 = [w for w in l.split('.') if w != 'www']
        l = ' '.join(l1)
        list_new.append(l)
    return list_new

def get_desc(sentence):
    tknzr = TweetTokenizer(strip_handles = True,reduce_len=True)
    sen=tokennizer_desc(tknzr,sentence)
    l1=[w for w in lemmatize((tknzr.tokenize(sen))) if w!='']
    return(' '.join(l1))
