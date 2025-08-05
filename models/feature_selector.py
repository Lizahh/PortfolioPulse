# models/feature_selector.py
from sklearn.feature_selection import SelectKBest, mutual_info_regression

def select_features(df, target_column, k=5):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_new, selected_features
