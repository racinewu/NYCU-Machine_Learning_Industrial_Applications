from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from nycu_midterm import read_txt

X, y = read_txt("wave60_dataset.txt", 60)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

knn_values = [1, 3, 5, 7, 9]


def evaluate_knn(weight_type):
    print(f"\n------- Weights: {weight_type} -------")
    for k in knn_values:
        model = KNeighborsRegressor(n_neighbors=k, weights=weight_type)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(
            f"KNN={k:<2} | Train: {train_score:.2f} | Test: {test_score:.2f}")


evaluate_knn("uniform")
evaluate_knn("distance")
