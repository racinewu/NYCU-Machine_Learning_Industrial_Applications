with open("prime_found.txt", "r") as f:
    content = f.read()

primes = [int(n) for n in content.split()]

count = sum(3000 < p < 6000 for p in primes)

print(f"I found {count} prime numbers between 3000 and 6000")
