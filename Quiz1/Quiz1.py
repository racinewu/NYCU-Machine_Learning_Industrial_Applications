import numpy as np
import math


def rotate(points, angle_deg):
    theta = math.radians(angle_deg)
    rotation_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                                [math.sin(theta),
                                 math.cos(theta)]])
    return np.dot(points, rotation_matrix.T)  # (N x 2) @ (2 x 2).T → (N x 2)


def read_data(filepath):
    coords = []
    with open(filepath, "r") as f:
        for _ in range(5):
            f.read(2)  # Ignore "[x,y]"
            x = float(f.read(3))
            y = float(f.read(3))
            coords.append((x, y))
            f.readline()
        f.read(2)
        x = float(f.read(4))
        f.read(3)
        y = float(f.read(4))
        coords.append((x, y))
    return np.array(coords)


def main():
    filename = "data1.txt"
    points = read_data(filename)

    angle = int(input("Enter the rotation angle: "))
    rotated = rotate(points, angle)

    # Sort according to x
    sorted_points = rotated[np.argsort(rotated[:, 0])]

    np.savetxt("frame0.csv", sorted_points, fmt="%.3f", delimiter=",")
    print("Saved to frame0.csv")


if __name__ == "__main__":
    main()
