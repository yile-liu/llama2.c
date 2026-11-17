// Naive GPU port of matmul() from run.c.
// Every call allocates device memory, copies x and w to the GPU, runs the
// kernel, copies xout back and frees the memory. Used only to verify correctness.

#include <cuda_runtime.h>

#define TPB 128

// W (d,n) @ x (n,) -> xout (d,), one thread per output element
__global__ void matmul_kernel(float* xout, const float* x, const float* w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
        val += w[i * n + j] * x[j];
    }
    xout[i] = val;
}

extern "C" void matmul(float* xout, float* x, float* w, int n, int d) {
    float *d_xout, *d_x, *d_w;
    cudaMalloc(&d_xout, d * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_w, (size_t)d * n * sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, (size_t)d * n * sizeof(float), cudaMemcpyHostToDevice);

    matmul_kernel<<<(d + TPB - 1) / TPB, TPB>>>(d_xout, d_x, d_w, n, d);

    cudaMemcpy(xout, d_xout, d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_xout);
    cudaFree(d_x);
    cudaFree(d_w);
}
