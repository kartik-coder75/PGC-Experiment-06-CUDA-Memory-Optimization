# Experiment 6: Memory Optimization in CUDA

**Course:** Memory Systems and Synchronization (Lab Manual)
**Topic:** Coalesced vs uncoalesced access, naive matrix multiplication, execution-time comparison
**Language:** CUDA C/C++ 

---

## 1. Aim

To study how the global-memory access pattern of a CUDA kernel affects performance by:

1. comparing an **uncoalesced** and a **coalesced** (shared-memory) matrix transpose,
2. running a **naive matrix multiplication**, and
3. comparing the execution times measured on the GPU with CUDA events.

### Learning outcomes
- Explain memory coalescing and why it matters for global-memory bandwidth.
- Write and launch 2D CUDA kernels using `dim3` grids and blocks.
- Use shared memory to turn a strided access into a coalesced one.
- Time kernels with CUDA events; compute bandwidth (GB/s) and GFLOPS.
- Verify GPU results against a CPU reference.

---

## 2. Theory

### 2.1 Memory coalescing
Threads execute in groups of 32 called **warps**. If the 32 threads of a warp access neighbouring addresses, the hardware merges them into a few wide transactions (**coalesced**). If the addresses are far apart (large stride), many transactions are needed and most fetched bytes are wasted (**uncoalesced**).

| | Coalesced | Uncoalesced |
|---|---|---|
| Thread *i* accesses | `base + i` | `base + i × N` (stride N) |
| Transactions per warp | Few (1 to 4) | Up to 32 |
| Bandwidth used | Close to peak | Small fraction of peak |

### 2.2 Matrix transpose
`out[j][i] = in[i][j]`; in row-major storage element (r, c) is at `r*N + c`.

- **Uncoalesced kernel:** reads a row (good) but writes a column with stride N (bad).
- **Coalesced kernel:** loads a 32×32 tile into shared memory (coalesced read), calls `__syncthreads()`, then writes the tile out transposed (coalesced write). The tile is declared `[32][33]`; the extra column avoids **shared-memory bank conflicts**.

### 2.3 Naive matrix multiplication
`C[i][j] = Σ_k A[i][k] · B[k][j]`. One thread computes one element of C with an N-iteration loop (2 global reads + 1 multiply-add per iteration). No data is reused via shared memory, so the kernel is memory-bound.

### 2.4 Formulas

| Quantity | Formula |
|---|---|
| Effective bandwidth (transpose) | `2 × N × N × 4 bytes / time` |
| Floating-point rate (matmul) | `2 × N³ / time` |
| Speedup | `t_uncoalesced / t_coalesced` |

---

## 3. Architecture / Execution model

### 3.1 Access patterns inside a warp

```
UNCOALESCED WRITE (transposeNaive)            COALESCED WRITE (transposeCoalesced)
thread 0 -> out[0*N + y]                      thread 0 -> out[y*N + x0 + 0]
thread 1 -> out[1*N + y]   (stride N)         thread 1 -> out[y*N + x0 + 1]   (stride 1)
thread 2 -> out[2*N + y]                      thread 2 -> out[y*N + x0 + 2]
   ...  up to 32 transactions                    ...  1 to 4 transactions
```

### 3.2 Coalesced transpose data flow (one 32×32 tile)

```
 Global memory (in)                 Shared memory tile[32][33]           Global memory (out)
 +----------------+  coalesced     +------------------+  coalesced      +----------------+
 | tile (bx, by)  | -------------> | tile[ty][tx]     | --------------> | tile (by, bx)  |
 +----------------+   read         +------------------+  write, reading +----------------+
                                      __syncthreads()     tile[tx][ty]
                                                          (block coords swapped)
```

### 3.3 Thread / grid mapping

| Kernel | Block | Grid | Thread owns |
|---|---|---|---|
| `transposeNaive`, `transposeCoalesced` | 32 × 32 (`TILE`) | ⌈N/32⌉ × ⌈N/32⌉ | one matrix element |
| `matMulNaive` | 16 × 16 (`BLK`) | ⌈N/16⌉ × ⌈N/16⌉ | one element of C |

### 3.4 Naive matmul memory traffic

```
Thread (row, col):   for k in 0..N-1:  sum += A[row][k] * B[k][col]
                                              ^ global read   ^ global read
=> ~2N³ global loads for 2N³ FLOPs  (≈ 1 load per FLOP, no reuse)
```

---

## 4. Requirements and setup

- NVIDIA GPU, NVIDIA driver, CUDA Toolkit (`nvcc`). On Windows, Visual Studio C++ build tools and the **x64 Native Tools Command Prompt** are needed.
- Alternatively use Google Colab with a GPU runtime (`!nvidia-smi` to confirm; `%%writefile exp6.cu` to create the file).

Check the tools:

```bash
nvidia-smi        # GPU name, driver, CUDA version
nvcc --version    # CUDA compiler version
```

**Environment used in this run**

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Compute capability | 12.0 (`sm_120`) |
| CUDA Toolkit | 13.0 (V13.0.48) |
| OS / shell | Windows, x64 Native Tools Command Prompt |
| Working folder | `C:\Users\User\cuda_exp6` |

> Note: `code` (VS Code) was not available on this machine, so the file was created and edited with `notepad exp6.cu`, and checked with `dir exp6.cu` and `powershell -Command "Get-Content exp6.cu -Tail 15"`.

---

## 5. Procedure

Create one file, `exp6.cu`, and paste Steps 1 to 5 into it in order.

### Step 1: Headers and error-check macro

```cuda
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32   // transpose block = 32 x 32 threads
#define BLK  16   // matmul block   = 16 x 16 threads

#define CHECK(call) do { cudaError_t e = (call);                      \
    if (e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",                   \
                cudaGetErrorString(e), __FILE__, __LINE__);           \
        exit(1); } } while (0)
```

### Step 2: Transpose kernels

```cuda
// Kernel 1: UNCOALESCED transpose
__global__ void transposeNaive(const float *in, float *out, int N)
{
    int x = blockIdx.x * TILE + threadIdx.x;   // column index
    int y = blockIdx.y * TILE + threadIdx.y;   // row index
    if (x < N && y < N)
        out[x * N + y] = in[y * N + x];   // read: row-wise (good)
                                          // write: column-wise, stride N (bad)
}

// Kernel 2: COALESCED transpose (shared-memory tile)
__global__ void transposeCoalesced(const float *in, float *out, int N)
{
    __shared__ float tile[TILE][TILE + 1];     // +1 padding avoids bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];   // coalesced read

    __syncthreads();                           // wait until the tile is loaded

    x = blockIdx.y * TILE + threadIdx.x;       // swap block coordinates
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];  // coalesced write
}
```

### Step 3: Naive matrix multiplication kernel

```cuda
__global__ void matMulNaive(const float *A, const float *B, float *C, int N)
{
    int row = blockIdx.y * BLK + threadIdx.y;
    int col = blockIdx.x * BLK + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)             // 2 global reads + 1 multiply-add
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
```

### Step 4: Host helpers 

Kernel launches are asynchronous, so **CUDA events** are used for timing. The first launch is a warm-up (not timed); the transpose is repeated 20 times and averaged.

```cuda
typedef void (*TKernel)(const float *, float *, int);

float timeTranspose(TKernel k, dim3 g, dim3 b, const float *in, float *out,
                    int N, int reps)
{
    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    CHECK(cudaMemset(out, 0, (size_t)N * N * sizeof(float)));

    k<<<g, b>>>(in, out, N);                       // warm-up launch (not timed)
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(s));
    for (int r = 0; r < reps; r++) k<<<g, b>>>(in, out, N);
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));

    float ms; CHECK(cudaEventElapsedTime(&ms, s, e));
    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    return ms / reps;                              // average time per launch
}

bool checkTranspose(const float *in, const float *out, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (out[j * N + i] != in[i * N + j]) return false;
    return true;
}

bool checkMatMul(const float *A, const float *B, const float *C, int N)
{
    for (int t = 0; t < 20; t++) {                 // spot-check 20 random elements
        int i = rand() % N, j = rand() % N;
        double ref = 0.0;
        for (int k = 0; k < N; k++) ref += (double)A[i * N + k] * B[k * N + j];
        if (fabs(ref - C[i * N + j]) > 1e-3 * fabs(ref) + 1e-3) return false;
    }
    return true;
}
```

### Step 5: `main()`

`main` allocates host/device memory, copies input, launches and times each kernel, copies results back, verifies them, and prints the table. Defaults: **N = 4096** (transpose) and **N = 1024** (matmul); both can be overridden on the command line.

```cuda
int main(int argc, char **argv)
{
    int NT = 4096;                 // transpose size (NT x NT)
    int NM = 1024;                 // matmul size    (NM x NM)
    if (argc > 1) NT = atoi(argv[1]);
    if (argc > 2) NM = atoi(argv[2]);

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);

    // ===== PART A : TRANSPOSE =====
    size_t bytesT = (size_t)NT * NT * sizeof(float);
    float *h_in  = (float *)malloc(bytesT);
    float *h_out = (float *)malloc(bytesT);
    for (size_t i = 0; i < (size_t)NT * NT; i++) h_in[i] = (float)(i % 1000);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  bytesT));
    CHECK(cudaMalloc(&d_out, bytesT));
    CHECK(cudaMemcpy(d_in, h_in, bytesT, cudaMemcpyHostToDevice));

    dim3 blockT(TILE, TILE);
    dim3 gridT((NT + TILE - 1) / TILE, (NT + TILE - 1) / TILE);

    float tN = timeTranspose(transposeNaive, gridT, blockT, d_in, d_out, NT, 20);
    CHECK(cudaMemcpy(h_out, d_out, bytesT, cudaMemcpyDeviceToHost));
    bool okN = checkTranspose(h_in, h_out, NT);

    float tC = timeTranspose(transposeCoalesced, gridT, blockT, d_in, d_out, NT, 20);
    CHECK(cudaMemcpy(h_out, d_out, bytesT, cudaMemcpyDeviceToHost));
    bool okC = checkTranspose(h_in, h_out, NT);

    double gbN = 2.0 * bytesT / (tN * 1e6);        // read + write, GB/s
    double gbC = 2.0 * bytesT / (tC * 1e6);
    printf("\n=== PART A: Matrix transpose (N = %d) ===\n", NT);
    printf("Uncoalesced : %8.3f ms  %7.1f GB/s  %s\n", tN, gbN, okN ? "PASS" : "FAIL");
    printf("Coalesced   : %8.3f ms  %7.1f GB/s  %s\n", tC, gbC, okC ? "PASS" : "FAIL");
    printf("Speedup (uncoalesced / coalesced) = %.2fx\n", tN / tC);

    // ===== PART B : NAIVE MATMUL =====
    size_t bytesM = (size_t)NM * NM * sizeof(float);
    float *hA = (float *)malloc(bytesM);
    float *hB = (float *)malloc(bytesM);
    float *hC = (float *)malloc(bytesM);
    for (size_t i = 0; i < (size_t)NM * NM; i++) {
        hA[i] = rand() / (float)RAND_MAX;
        hB[i] = rand() / (float)RAND_MAX;
    }
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytesM)); CHECK(cudaMalloc(&dB, bytesM)); CHECK(cudaMalloc(&dC, bytesM));
    CHECK(cudaMemcpy(dA, hA, bytesM, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytesM, cudaMemcpyHostToDevice));

    dim3 block(BLK, BLK);
    dim3 grid((NM + BLK - 1) / BLK, (NM + BLK - 1) / BLK);

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    matMulNaive<<<grid, block>>>(dA, dB, dC, NM);  // warm-up
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());

    int reps = 5;
    CHECK(cudaEventRecord(s));
    for (int r = 0; r < reps; r++) matMulNaive<<<grid, block>>>(dA, dB, dC, NM);
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    float ms; CHECK(cudaEventElapsedTime(&ms, s, e));
    float tM = ms / reps;

    CHECK(cudaMemcpy(hC, dC, bytesM, cudaMemcpyDeviceToHost));
    bool okM = checkMatMul(hA, hB, hC, NM);
    double gflops = 2.0 * NM * NM * (double)NM / (tM * 1e6);
    printf("\n=== PART B: Naive matrix multiplication (N = %d) ===\n", NM);
    printf("Naive matmul: %8.3f ms  %7.1f GFLOPS  %s\n", tM, gflops, okM ? "PASS" : "FAIL");

    // ===== SUMMARY =====
    printf("\n=== SUMMARY ===\n");
    printf("Transpose uncoalesced : %.3f ms\n", tN);
    printf("Transpose coalesced   : %.3f ms\n", tC);
    printf("Naive matmul          : %.3f ms\n", tM);

    cudaFree(d_in); cudaFree(d_out); cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(h_in); free(h_out); free(hA); free(hB); free(hC);
    return 0;
}
```

### Step 6: Compile

```bash
nvcc -O2 -o exp6 exp6.cu                       # Linux / Colab
nvcc -O2 -o exp6.exe exp6.cu                   # Windows (Developer prompt)
nvcc -O2 -arch=sm_120 -o exp6.exe exp6.cu      # used here (RTX 5060 Ti, cc 12.0)
```

Other `-arch` values: `sm_75` (Colab T4), `sm_86` (RTX 30-series).

### Step 7: Run and record

```bash
exp6.exe              # Windows, default sizes: transpose 4096, matmul 1024
./exp6                # Linux / Colab
./exp6 2048 512       # optional: transpose N=2048, matmul N=512
```

Run at least **three** times and record the values.

---

## 6. Results

Compile command and three runs on the RTX 5060 Ti:

### 6.1 Raw readings (3 runs)

| Run | Uncoalesced (ms) | Uncoalesced (GB/s) | Coalesced (ms) | Coalesced (GB/s) | Transpose speedup | Naive matmul (ms) | Naive matmul (GFLOPS) | Checks |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.804 | 167.0 | 0.416 | 322.9 | 1.93x | 1.374 | 1563.3 | all PASS |
| 2 | 0.804 | 167.0 | 0.414 | 324.3 | 1.94x | 1.373 | 1563.7 | all PASS |
| 3 | 0.805 | 166.7 | 0.412 | 325.6 | 1.95x | 1.544 | 1390.7 | all PASS |
| **Average** | **0.804** | **166.9** | **0.414** | **324.3** | **1.94x** | **1.430** | **1505.9** | PASS |

### 6.2 Observation table

| Kernel | Matrix size N | Time (ms) | Speedup | Bandwidth (GB/s) / GFLOPS | Result (PASS/FAIL) |
|---|---|---|---|---|---|
| Transpose: uncoalesced | 4096 | 0.804 | 1.0x (baseline) | 166.9 GB/s | PASS |
| Transpose: coalesced | 4096 | 0.414 | 0.804 / 0.414 = **1.94x** | 324.3 GB/s | PASS |
| Naive matrix multiplication | 1024 | 1.430 | - | 1505.9 GFLOPS | PASS |

**Speedup of coalesced over uncoalesced transpose = 1.94 x**

**GPU model used:** NVIDIA GeForce RTX 5060 Ti  **Compute capability:** 12.0

(Averages are over the three runs shown above; for example, matmul average = (1.374 + 1.373 + 1.544) / 3 = 1.430 ms.)

---

## 7. Analysis and discussion

- **Why the coalesced transpose wins.** In the uncoalesced kernel every warp writes 32 floats that are N elements apart, so one warp-write splits into many memory transactions. In the coalesced kernel, the shared-memory tile absorbs the transposition so both the global read and write are contiguous. Result: effective bandwidth rose from about **167 GB/s to about 324 GB/s** (≈ 1.94x).
- **Why the speedup is only about 2x.** The sample output in the manual (a Tesla T4) shows 2.57x. Newer GPUs have larger L2 caches and better write handling, which soften the penalty of strided writes, so the gap is smaller here. The pattern (both PASS, coalesced clearly faster) still matches the expected result.
- **Padding (`TILE + 1`).** Without the extra column, threads of a warp reading a column of the tile hit the same shared-memory bank (a bank conflict). Padding spreads them across banks. Optional exercise: remove the `+1`, recompile and compare the time.
- **Naive matmul.** About 2N³ global loads for 2N³ FLOPs (one load per FLOP) with no reuse. It reached roughly 1.4 to 1.56 TFLOPS, only a small fraction of the GPU's peak FP32 throughput. The figure is higher than the manual's T4 example because a 1024×1024 float matrix is only 4 MB, so A and B largely stay in the large L2 cache. Shared-memory tiling (tile sizes 8, 16, 32) is the next optimisation.
- **Run-to-run variation.** Transpose times were very stable (0.804 to 0.805 ms and 0.412 to 0.416 ms). Matmul run 3 was slower (1.544 ms vs about 1.374 ms), likely due to clock or power-state variation or background GPU activity on a desktop system, which is why repeated runs are recommended.
- **Timing method.** CUDA events measure time on the GPU itself; a warm-up launch plus averaging over repeats removes start-up cost and noise.

---

## 8. Conclusion

The same amount of data movement ran at very different speeds depending on the access pattern. Using a shared-memory tile to make both global read and write coalesced improved the 4096×4096 transpose from 0.804 ms (166.9 GB/s) to 0.414 ms (324.3 GB/s), a **1.94x speedup**. The naive matrix multiplication (1024×1024) ran in about 1.43 ms (about 1506 GFLOPS) and passed verification, confirming that a kernel with no data reuse stays limited by global-memory traffic and is the baseline for shared-memory tiling.

---

## 9. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `nvcc: command not found` | CUDA Toolkit not installed or not on PATH. On Windows use the x64 Native Tools / Developer prompt; on Colab use a GPU runtime. |
| `'code' is not recognized` | VS Code command not on PATH; use `notepad exp6.cu` instead (as done here). |
| `no CUDA-capable device is detected` | No NVIDIA GPU or driver missing. Check `nvidia-smi`; on Colab enable the GPU runtime. |
| Result shows `FAIL` | Check index expressions, especially the swapped block indices in the coalesced kernel and bounds checks. |
| Very small speedup | Matrix too small. Use N = 4096 or larger so memory traffic dominates. |
| Out of memory | Reduce N, e.g. `exp6.exe 2048 512`. |

---

## 10. Repository layout

```

CUDA-Experiment-06-Memory-Optimization/
├── .gitignore
├── README.md
├──screenshots/
│   └── 01_nvcc_version_and_coalesced_transpose_kernel.png
│   └── 02_naive_matmul_kernel_and_cpu_verification.png
│   └── 03_main_function_summary_and_memory_cleanup.png
│   └── 04_nvcc_compilation_and_rtx5060ti_benchmark_runs.png
├── src/
│   └── exp6.cu
└── results/
    └── benchmark_output.txt

```
