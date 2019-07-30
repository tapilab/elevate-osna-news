import glob


def read_data(directory):
    dfs = []
    for label in ['real', 'fake']:
        for file in glob.glob(directory + os.path.sep + label + os.path.sep + '*gz'):
            print('reading %s' % file)
            df = pd.DataFrame((json.loads(line) for line in gzip.open(file)))
            df['label'] = label
            dfs.append(df)
    return pd.concat(dfs)[['publish_date', 'source', 'text', 'title', 'tweets', 'label']]
