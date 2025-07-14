from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

data = pd.read_csv("hw5_cancer.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=0)

# model param
models = {
    "DecisionTree": {
        "model":
        DecisionTreeClassifier(criterion="entropy",
                               splitter="best",
                               max_depth=5,
                               max_leaf_nodes=7),
        "params": [
            "criterion = entropy", "splitter = best", "max_depth = 5",
            "max_leaf_nodes = 7"
        ]
    },
    "RandomForest": {
        "model":
        RandomForestClassifier(criterion="entropy",
                               n_estimators=80,
                               max_features=7,
                               max_depth=3),
        "params": [
            "criterion = entropy", "n_estimators = 80", "max_features = 7",
            "max_depth = 3"
        ]
    },
    "GradientBoost": {
        "model":
        GradientBoostingClassifier(max_depth=3,
                                   n_estimators=30,
                                   learning_rate=0.03),
        "params":
        ["max_depth = 3", "n_estimators = 30", "learning_rate = 0.03"]
    }
}

results = {}

for name, content in models.items():
    clf = content["model"]
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    results[name] = {
        "train_score": round(train_score, 3),
        "test_score": round(test_score, 3)
    }

param_dict = {name: content["params"] for name, content in models.items()}
print(yaml.dump(param_dict, sort_keys=False, default_flow_style=False))

for name, res in results.items():
    print(
        f"The train/test score of {name} is {res['train_score']:.3f}/{res['test_score']:.3f}"
    )
