def time_split(df, split_date):
    train = df[df['date'] < split_date]
    test = df[df['date'] >= split_date]
    return train, test