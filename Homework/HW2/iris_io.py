import numpy as np


def parse_iris_data(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # 去除空行與空白

    inputs = []
    targets = []

    # parse inputs
    input_started = False
    target_started = False
    for line in lines:
        if line.startswith("all inputs:"):
            input_started = True
            continue
        elif line.startswith("all targets:"):
            input_started = False
            target_started = True
            continue

        if input_started and line.startswith("["):
            values = list(map(float, line.strip("[]").split()))
            inputs.append(values)
        elif target_started:
            targets.extend(map(int, line.split()))

    return np.array(inputs), np.array(targets)


def main():
    input_path = "iris_dataset.txt"
    output_path = "iris_data.csv"

    X, y = parse_iris_data(input_path)

    # calculate median
    medians = np.median(X, axis=0)
    print("The median values of four features:",
          ', '.join(f"{m:.2f}" for m in medians))

    # count num
    count0 = np.sum(y == 0)
    count1 = np.sum(y == 1)
    count2 = np.sum(y == 2)
    print(
        f"The numbers for each species is Setosa : {count0}, versicolor : {count1}, virginica : {count2}"
    )

    data = np.hstack((X, y.reshape(-1, 1)))
    np.random.shuffle(data)

    np.savetxt(output_path, data, fmt="%.1f", delimiter=",")


if __name__ == "__main__":
    main()
