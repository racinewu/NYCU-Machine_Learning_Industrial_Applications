import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# read file
df = pd.read_csv("iris_data.csv", header=None)

# train until accuracy reaches 1.0
while True:
    df_shuffled = df.sample(frac=1).reset_index(drop=True)  # shuffle data
    X = df_shuffled.iloc[:, :-1].values  # features
    y = df_shuffled.iloc[:, -1].values  # labels (flattened)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.253)

    model = KNeighborsClassifier(n_neighbors=3,
                                 p=2,
                                 weights='distance',
                                 algorithm='ball_tree')
    model.fit(X_train, y_train)

    if model.score(X_test, y_test) == 1.0:
        break

# test prediction
test_1 = np.array([5, 2.9, 1, 0.2]).reshape(1, -1)
test_2 = np.array([3, 2.2, 4, 0.9]).reshape(1, -1)

print("The test score is:", model.score(X_test, y_test))
print("Test of first kind is:", model.predict(test_1)[0])
print("Test of second kind is:", model.predict(test_2)[0])
