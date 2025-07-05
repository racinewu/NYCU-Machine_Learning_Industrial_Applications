import cmath


def solve_quadratic(a, b, c):
    D = b**2 - 4 * a * c
    sqrt_D = cmath.sqrt(D)
    x1 = (-b + sqrt_D) / (2 * a)
    x2 = (-b - sqrt_D) / (2 * a)

    if D > 0:
        print(f"x1 = {x1.real:.2f}, x2 = {x2.real:.2f}, 2 real roots")
    elif D < 0:
        print(
            f"x1 = {x1.real:.2f} + ({x1.imag:.2f})J, x2 = {x2.real:.2f} - ({-x2.imag:.2f})J"
        )
    else:
        print(f"x1 = x2 = {x1.real:.2f}, same real root")


def main():
    while True:
        print("Solve 2nd order equation (a X^2 + b X + c = 0)")
        try:
            a = float(input("Enter a = "))
            if a == 0:
                print("Exit loop.")
                break
            b = float(input("Enter b = "))
            c = float(input("Enter c = "))
            solve_quadratic(a, b, c)
            print()
        except ValueError:
            print("Invalid input. Please enter numeric values.\n")


if __name__ == "__main__":
    main()
