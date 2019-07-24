import pandas as pd
from collections import Counter
import re

def Mystats(directory):
    df=pd.read_csv(directory)
    id=df['social_id'].unique()
    #1
    print('Q1:Number of unique users:',len(id))
    mes=df['comment_tokens']
    #2
    print('Q2:Number of unique messages:',len(mes.unique()))
    #4

    word=[]
    for m in mes.astype(str):
        mes=m.split()
        for mes1 in mes:
            mes1=re.sub("[0-9\W+]","",mes1)
            # print(mes1)
            if(mes1!=""):
                word.append(mes1)

    word1=list(set(word))
    print('Q4:Number of unique words:',len(word1))
    #5
    print('Q5:Number of tokens:', len(mes))
    #6
    c=Counter(word)
    print('Q6:50 most common words:',c.most_common(50))

    word1 = []
    df1=pd.read_csv('D:\\news\\training_data\\factchecks.csv')
    true=df1[(df1.site=='youtube')&(df1.ruling=='TRUE')]
    msgtrue=true['social_id']
    print('Q3:Number of users/message in class TRUE:', len(msgtrue))

    pd1=pd.merge(df,true,on=['social_id','site'],how='inner')
    word1=tweet_tokenizer(pd1)
    print('Q7:50 most common words:', Counter(word1).most_common(50))

    false = df1[(df1.site == 'youtube') & (df1.ruling == 'FALSE')]
    msgfalse = false['social_id']
    print('Q3:Number of users/message in class FALSE:', len(msgfalse))
    pd1 = pd.merge(df, false, on=['social_id', 'site'], how='inner')
    word1 = tweet_tokenizer(pd1)
    print('Q7:50 most common words:', Counter(word1).most_common(50))

    fire= df1[(df1.site == 'youtube') & (df1.ruling == 'Pants on Fire!')]
    msgfire = fire['social_id']
    print('Number of users/message in class Pants on Fire:', len(msgfire))
    pd1 = pd.merge(df, fire, on=['social_id', 'site'], how='inner')
    word1 = tweet_tokenizer(pd1)
    print('Q7:50 most common words:', Counter(word1).most_common(50))

    mt = df1[(df1.site == 'youtube') & (df1.ruling == 'Mostly True')]
    msgmt = mt['social_id']
    print('Number of users/message in class Mostly True:', len(msgmt))
    pd1 = pd.merge(df, mt, on=['social_id', 'site'], how='inner')
    word1 = tweet_tokenizer(pd1)
    print('Q7:50 most common words:', Counter(word1).most_common(50))

    mf = df1[(df1.site == 'youtube') & (df1.ruling == 'Mostly False')]
    msgmf = mf['social_id']
    print('Number of users/message in class Mostly False:', len(msgmf))
    pd1 = pd.merge(df, mf, on=['social_id', 'site'], how='inner')
    word1 = tweet_tokenizer(pd1)
    print('Q7:50 most common words:', Counter(word1).most_common(50))

    ht = df1[(df1.site == 'youtube') & (df1.ruling == 'Half-True')]
    msgfire = ht['social_id']
    print('Number of users/message in class Half-True:', len(ht))
    pd1 = pd.merge(df, ht, on=['social_id', 'site'], how='inner')
    word1 = tweet_tokenizer(pd1)
    print('Q7:50 most common words:', Counter(word1).most_common(50))

    mx = df1[(df1.site == 'youtube') & (df1.ruling == 'MIXTURE')]
    msgfire = mx['social_id']
    print('Number of users/message in class MIXTURE:', len(mx))
    pd1 = pd.merge(df, mx, on=['social_id', 'site'], how='inner')
    word1 = tweet_tokenizer(pd1)
    print('Q7:50 most common words:', Counter(word1).most_common(50))


def tweet_tokenizer(df):
    list=[]
    msg = df['comment_tokens']
    for m in msg.astype(str):
        mes = m.split()
        for mes1 in mes:
            mes1 = re.sub("[0-9\W+]", "", mes1)
            if (mes1!= ""):
                list.append(mes1)
    print(list)
    return list




if __name__=='__main__':
    Mystats(directory)
