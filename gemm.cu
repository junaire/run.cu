#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

void checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error in file " << __FILE__ << " at line " << __LINE__
              << ": " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void checkCublasError(cublasStatus_t stat) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error in file " << __FILE__ << " at line " << __LINE__
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  int M = 1024;
  int K = 512;
  int N = 256;

  if (argc == 4) {
    M = std::atoi(argv[1]);
    K = std::atoi(argv[2]);
    N = std::atoi(argv[3]);
  } else {
    std::cout << "Using default matrix dimensions: " << M << " x " << K << " x "
              << N << std::endl;
  }

  float *h_A = (float *)malloc(M * K * sizeof(float));
  float *h_B = (float *)malloc(K * N * sizeof(float));
  float *h_C = (float *)malloc(M * N * sizeof(float));

  for (int i = 0; i < M * K; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < K * N; ++i) {
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float *d_A, *d_B, *d_C;

  checkCudaError(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
  checkCudaError(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
  checkCudaError(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

  checkCudaError(
      cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaError(
      cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  checkCublasError(cublasCreate(&handle));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cudaEvent_t start, stop;
  checkCudaError(cudaEventCreate(&start));
  checkCudaError(cudaEventCreate(&stop));

  checkCudaError(cudaEventRecord(start));
  checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                               &alpha, d_A, M, d_B, K, &beta, d_C, M));
  checkCudaError(cudaEventRecord(stop));

  checkCudaError(cudaEventSynchronize(stop));

  float milliseconds = 0;
  checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop));

  checkCudaError(
      cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  std::cout << "Result C[0][0]: " << h_C[0] << std::endl;
  std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  checkCublasError(cublasDestroy(handle));

  checkCudaError(cudaEventDestroy(start));
  checkCudaError(cudaEventDestroy(stop));

  return 0;
}
