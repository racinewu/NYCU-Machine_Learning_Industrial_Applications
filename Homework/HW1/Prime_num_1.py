import math


def is_prime(x):
    if x < 2:
        return False
    for i in range(2, int(math.sqrt(x)) + 1):
        if x % i == 0:
            return False
    return True


primes = [i for i in range(100, 9001) if is_prime(i)][::-1]

with open("prime_found.txt", "w") as f:
    for i in range(0, len(primes), 6):
        line = ' '.join(str(p) for p in primes[i:i + 6])
        f.write(line + "\n")
