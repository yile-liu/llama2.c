// GPU port of matmul() from run.c with the weights resident in GPU memory.
// The whole checkpoint is copied to the GPU once in matmul_init(), and the
// buffers for x and xout are allocated once there too. Each matmul() call only
// maps the host weight pointer to the device copy and transfers x and xout.

#include <cuda_runtime.h>

#define TPB 128
#define MAX_DIM 32000 // largest n or d: vocab_size of the classifier

static float* h_data = NULL; // host checkpoint (mmap)
static float* d_data = NULL; // device copy of the checkpoint
static float* d_x    = NULL; // input buffer, allocated once
static float* d_xout = NULL; // output buffer, allocated once

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

// copy all weights to the GPU once and allocate the x / xout buffers
extern "C" void matmul_init(float* data, size_t bytes) {
    h_data = data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_x, MAX_DIM * sizeof(float));
    cudaMalloc(&d_xout, MAX_DIM * sizeof(float));
}

extern "C" void matmul_free(void) {
    cudaFree(d_data);
    cudaFree(d_x);
    cudaFree(d_xout);
}

extern "C" void matmul(float* xout, float* x, float* w, int n, int d) {
    float* d_w = d_data + (w - h_data); // same offset in the device copy

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    matmul_kernel<<<(d + TPB - 1) / TPB, TPB>>>(d_xout, d_x, d_w, n, d);

    cudaMemcpy(xout, d_xout, d * sizeof(float), cudaMemcpyDeviceToHost);
}
