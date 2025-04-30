import numpy as np
import time
from multiprocessing import Pool, shared_memory
import ctypes

SIZE = 2048
BLOCK_SIZE = 128  
WORKERS = 8       

def init_worker(shared_a, shared_b, shape, dtype):
    global A_shmem, B_shmem
    A_shmem = shared_memory.SharedMemory(name=shared_a)
    B_shmem = shared_memory.SharedMemory(name=shared_b)
    global A, B
    A = np.ndarray(shape, dtype=dtype, buffer=A_shmem.buf)
    B = np.ndarray(shape, dtype=dtype, buffer=B_shmem.buf)
def block_multiply(args):
    i_start = args
    n = A.shape[0]
    block = np.zeros((BLOCK_SIZE, n), dtype=np.complex128)
    for k in range(0, n, BLOCK_SIZE):
        a_block = A[i_start:i_start+BLOCK_SIZE, k:k+BLOCK_SIZE]
        for j in range(0, n, BLOCK_SIZE):
            block[:, j:j+BLOCK_SIZE] += a_block @ B[k:k+BLOCK_SIZE, j:j+BLOCK_SIZE]
    return i_start, block
def parallel_matmul_optimized(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.complex128)
    shm_a = shared_memory.SharedMemory(create=True, size=A.nbytes)
    shm_b = shared_memory.SharedMemory(create=True, size=B.nbytes)
    A_shared = np.ndarray(A.shape, dtype=A.dtype, buffer=shm_a.buf)
    B_shared = np.ndarray(B.shape, dtype=B.dtype, buffer=shm_b.buf)
    np.copyto(A_shared, A)
    np.copyto(B_shared, B)
    with Pool(WORKERS, initializer=init_worker, 
             initargs=(shm_a.name, shm_b.name, A.shape, A.dtype)) as pool:
        results = pool.map(block_multiply, range(0, n, BLOCK_SIZE))
    for i, block in results:
        C[i:i+BLOCK_SIZE, :] = block
    shm_a.close()
    shm_b.close()
    shm_a.unlink()
    shm_b.unlink()
    return C
def benchmark():
    np.random.seed(42)
    A = (np.random.rand(SIZE, SIZE) + 1j*np.random.rand(SIZE, SIZE)).astype(np.complex128)
    B = (np.random.rand(SIZE, SIZE) + 1j*np.random.rand(SIZE, SIZE)).astype(np.complex128)
    _ = parallel_matmul_optimized(A[:256, :256], B[:256, :256])
    start = time.perf_counter()
    C = parallel_matmul_optimized(A, B)
    elapsed = time.perf_counter() - start
    ref = A @ B
    assert np.allclose(C, ref, atol=1e-5), "Ошибка валидации"
    flops = 2 * SIZE**3
    perf = flops / elapsed * 1e-6
    print(f"Оптимизированный параллельный алгоритм:")
    print(f"Размер матриц: {SIZE}x{SIZE}")
    print(f"Затраченное время: {elapsed:.5f} сек")
    print(f"Производительность: {perf:.5f} MFlops")
    print("Ковалевская А.М., 020303-АИСа-о24")
if __name__ == "__main__":
    benchmark()
