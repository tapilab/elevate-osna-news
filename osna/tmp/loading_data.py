import glob,json
import os,gzip
import pandas as pd


def read_data(directory):
    dfs = []
    for label in ['real', 'fake']:
        for file in glob.glob(directory + os.path.sep + label + os.path.sep + '*gz'):
            print('reading %s' % file)
            df = pd.DataFrame((json.loads(line) for line in gzip.open(file)))
            df['label'] = label
            dfs.append(df)
    df=pd.concat(dfs)[['publish_date', 'source', 'text', 'title', 'tweets', 'label']]
    list_text = [i for i in list(df.text) if i != '']
    return df[df.text.isin(list_text)]
