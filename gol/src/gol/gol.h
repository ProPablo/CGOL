#pragma once

#include "cuda_runtime.h"

//bool* initCUDA(bool* host_A);
//http://www.cplusplus.com/doc/oldtutorial/templates/
template <class T>
T* initCUDA(T* host_A, int bufferSize)
{
	cudaError_t cudaStatus;
	T* device_A;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
		cudaFree(device_A);
		exit(cudaStatus);
	}

	cudaStatus = cudaMalloc((void**)&device_A, bufferSize);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaMalloc failed!";
		cudaFree(device_A);
		exit(cudaStatus);
	}

	cudaStatus = cudaMemcpy(device_A, host_A, bufferSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaMemcpy failed!";
		cudaFree(device_A);
		exit(cudaStatus);
	}
	return device_A;
}

template <class T>
void endCUDA(T* host_A, T* device_A, int bufferSize)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		cudaFree(device_A);
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy(host_A, device_A, bufferSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaMemcpy failed!";
		cudaFree(device_A);
		exit(cudaStatus);
	}
	cudaFree(device_A);
}


void runKernel(int gen, bool* device_A);
void runBP(int gen, uint32_t* device_A);
//void endCUDA(bool* host_A, bool* device_A);
void render(cudaGraphicsResource* cuda_pbo_resource, bool* prevGen, bool* currGen);
//void printDeviceMemory(void* deviceptr);