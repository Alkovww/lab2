import numpy as np
import time

def generate_matrix(n):
    return np.random.rand(n, n) + 1j * np.random.rand(n, n)

def naive_matmul(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

n = 2048
A = generate_matrix(n)
B = generate_matrix(n)

start = time.time()
C = naive_matmul(A, B)
end = time.time()

t = end - start
c = 2 * n ** 3
mflops = (c / t) * 1e-6
print("\nНаивный алгоритм:")
print(f"Размер матриц: {n} x {n}")
print(f"Затраченное время: {t:.5f} секунд")
print(f"Производительность: {mflops:.5f} MFlops")
print("Ковалевская А.М., 020303-АИСа-о24")
