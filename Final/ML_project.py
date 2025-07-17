import numpy as np
import pandas as pd
import yaml
from typing import Tuple
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score


def cross_validation(data: np.ndarray, split: int, fold_size: int,
                     num_features: int) -> Tuple[np.ndarray, ...]:
    start, end = split * fold_size, (split + 1) * fold_size
    test_data = data[start:end]
    train_data = np.concatenate((data[:start], data[end:]), axis=0)
    x_train, y_train = train_data[:, :num_features], train_data[:, -1]
    x_test, y_test = test_data[:, :num_features], test_data[:, -1]
    return x_train, y_train, x_test, y_test


def dump_hyperparams():
    param_text = {
        "DecisionTreeRegressor": {
            "max_depth": 49,
            "min_samples_split": 9,
            "min_samples_leaf": 9,
            "random_state": 0,
            "splitter": "best"
        },
        "RandomForestRegressor": {
            "n_estimators": 49,
            "max_depth": 15,
            "random_state": 0
        },
        "GradientBoostingRegressor": {
            "n_estimators": 400,
            "max_depth": 10,
            "learning_rate": 0.01,
            "min_samples_split": 60,
            "tol": 0.003,
            "random_state": 0
        }
    }
    print("Hyperparameters:\n" + yaml.dump(param_text, sort_keys=False))


def evaluate_model(name: str,
                   model,
                   data: np.ndarray,
                   num_features: int,
                   k: int = 5,
                   repeat: int = 5):
    scores_train, scores_test = [], []
    fold_size = len(data) // k

    for _ in range(repeat):
        np.random.shuffle(data)
        for split in range(k):
            x_train, y_train, x_test, y_test = cross_validation(
                data, split, fold_size, num_features)
            model.fit(x_train, y_train)
            scores_train.append(r2_score(y_train, model.predict(x_train)))
            scores_test.append(r2_score(y_test, model.predict(x_test)))

    print(
        f"{name} avg R²: train={np.mean(scores_train):.3f}, test={np.mean(scores_test):.3f}"
    )
    print(
        f"Standard deviation: train={np.std(scores_train):.3f}, test={np.std(scores_test):.3f}\n"
    )


if __name__ == "__main__":
    df = pd.read_csv("Top_600_IMDB_Movies_processed.csv")
    data = df.iloc[:, 1:].to_numpy()
    np.random.seed(0)
    np.random.shuffle(data)

    num_features = data.shape[1] - 1

    models = {
        "DecisionTreeRegressor":
        DecisionTreeRegressor(max_depth=49,
                              min_samples_split=9,
                              min_samples_leaf=9,
                              random_state=0,
                              splitter="best"),
        "RandomForestRegressor":
        RandomForestRegressor(n_estimators=49, max_depth=15, random_state=0),
        "GradientBoostingRegressor":
        GradientBoostingRegressor(n_estimators=400,
                                  max_depth=10,
                                  learning_rate=0.01,
                                  min_samples_split=60,
                                  tol=0.003,
                                  random_state=0)
    }

    dump_hyperparams()
    for name, model in models.items():
        evaluate_model(name, model, data, num_features)
