import pandas as pd

from osna.mytwitter import Twitter
from osna import credentials_path

# ages ={}
# ages['1'] = [1,2,3,4,5,6,7,8,9,1,2,10,20,30]
# ages['2'] = [1,2,3,4,5,6,7,8,9,1,2,10,20,30]
# # qcats1 = pd.cut(ages['1'], [0,6,8,10,31], labels=False)
# # print(qcats1)
#
# f = quantization(ages,[0,6,8,10,31])
# print(f)

t = Twitter(credentials_path)

tw = t._search_tweets('https://www.cnn.com/2019/08/02/politics/trump-ratcliffe-dni/index.html')
print(tw)
