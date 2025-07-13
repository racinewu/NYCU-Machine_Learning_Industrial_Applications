import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

data = pd.read_csv("hw4_boston.csv")

fold_size = 101
n_splits = 5
n_features = data.shape[1] - 1

for fold in range(n_splits):
    # Split training data (exclusive current fold)
    train_data = pd.concat(
        [data.iloc[:fold * fold_size], data.iloc[(fold + 1) * fold_size:]])

    x_train = train_data.iloc[:, :n_features].values
    y_train = train_data.iloc[:, n_features].values

    # Split training data (current fold)
    test_data = data.iloc[fold * fold_size:(fold + 1) * fold_size]
    x_test = test_data.iloc[:, :n_features].values
    y_test = test_data.iloc[:, n_features].values

    model = Ridge(alpha=0.1)
    model.fit(x_train, y_train)

    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    print(
        f"Ridge = 0.1 | Fold {fold + 1}: Train/Test Scores = {train_score:.2f}/{test_score:.2f}"
    )
