import numpy as np
import time
from scipy.linalg.blas import zgemm
MATRIX_SIZE = 2048
def generate_complex_matrix(size):
    """Генерация случайной комплексной матрицы заданного размера."""
    return np.random.rand(size, size) + 1j * np.random.rand(size, size)

if __name__ == "__main__":
    matrix_a = generate_complex_matrix(MATRIX_SIZE)
    matrix_b = generate_complex_matrix(MATRIX_SIZE)

    start_time = time.perf_counter()
    result_matrix = zgemm(alpha=1.0, a=matrix_a, b=matrix_b)
    end_time = time.perf_counter()

    operations_count = 2 * MATRIX_SIZE**3
    execution_time = end_time - start_time
    performance_mflops = operations_count / execution_time * 1e-6

    print("\nУмножение матриц с использованием BLAS (zgemm):")
    print(f"Размер матриц: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"Затраченное время: {execution_time:.5f} секунд")
    print(f"Производительность: {performance_mflops:.5f} MFlops")
    print("Ковалевская А.М., 020303-АИСа-о24")
