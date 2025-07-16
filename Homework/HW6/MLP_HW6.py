import pandas as pd
import yaml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv("hw6_haberman.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

mlp_params = {
    "hidden_layer_sizes": (7, 5, 6),
    "activation": "tanh",
    "solver": "lbfgs",
    "max_iter": 20000,
    "learning_rate_init": 0.55,
    "random_state": 0,
    "alpha": 0.85,
    "learning_rate": "adaptive",
    "warm_start": True
}

model = make_pipeline(StandardScaler(), MLPClassifier(**mlp_params))
model.fit(X, y)

train_score = model.score(X, y)

yaml_params = {
    "MLP": [f"{key} = {value}" for key, value in mlp_params.items()]
}

print(yaml.dump(yaml_params, sort_keys=False, default_flow_style=False))
print(f"MLP training set score: {train_score:.3f}")
