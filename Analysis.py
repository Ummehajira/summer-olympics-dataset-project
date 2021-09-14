import pandas as pd

def load_data(path):
    df=pd.read_csv(path, skiprows=1)
    col_rename = {'Unnamed: 0': 'Country',
                      '01 !': 'Gold',
                      '02 !': 'Silver',
                      '03 !': 'Bronze',
                      '01 !.1': 'Gold',
                      '02 !.1': 'Silver',
                      '03 !.1': 'Bronze',
                      '01 !.2': 'Gold',
                      '02 !.2': 'Silver',
                      '03 !.2': 'Bronze',
                      'Total.1': 'Total'}
    df.rename(columns=col_rename, inplace=True)
    country_names = [x.split('\xc2\xa0(')[0] for x in df.iloc[:, 0]]
    df.set_index(pd.Series(country_names), inplace=True)
    df.iloc[:, 0] = country_names
    df.drop('Total', axis=1, inplace=True)
    return df

def get_points(df):
    df['points'] = 3 * (df['Gold'].sum(axis=1)) + 2 * (df['Silver'].sum(axis=1)) + 1 * (df['Bronze'].sum(axis=1))
    return df.iloc[:, 14]

def kMeans(df):
    from sklearn.preprocessing import LabelEncoder
    def label_encoder(df, list_of_columns):

        for col in list_of_columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
        return df

    label_encoder(df, ['Country'])

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5, init='k-means++', n_init=10)
    km=km.fit(df)
    return km.cluster_centers_
