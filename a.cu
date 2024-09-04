#include <stdio.h>

#define N 1024

__global__ void vectorAdd(const float *A, const float *B, float *C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

int main(int argc, char **argv) {
  if (argc > 1) {
    printf("Received arguments: ");
  }
  for (int i = 1; i < argc; ++i) {
    printf("%s ", argv[i]);
  }
  printf("\n");

  float A[N], B[N], C[N];

  for (int i = 0; i < N; i++) {
    A[i] = i * 123.0f;
    B[i] = i * 2.0f;
  }

  float *d_A, *d_B, *d_C;

  cudaMalloc((void **)&d_A, N * sizeof(float));
  cudaMalloc((void **)&d_B, N * sizeof(float));
  cudaMalloc((void **)&d_C, N * sizeof(float));

  cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C);
  cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Result: \n");
  for (int i = 0; i < 10; i++) {
    printf("%f + %f = %f\n", A[i], B[i], C[i]);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
