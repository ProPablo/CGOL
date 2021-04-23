#include "cuda_runtime.h"
#include <iostream>
#include <stdlib.h>

#include "../Helpers.h"
#include "device_launch_parameters.h"

//__constant__ int warpSize;

//__constant__ int perlin_A[256];

#define FULL_MASK 0xffffffff

__constant__ int N_size;

int _N_size;

__global__ void CUDAgameLoop(bool* data, int currentGen)
{
	//Indexing similiar to 2d flattening
	//Treating blocks as 2nd dim (y), and thread as 1st dim (x)
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < N && x < N) {
		bool isAlive = data[((currentGen - 1) * N + y) * N + x];
		unsigned int totalNeighbours = 0;
		for (int i = (y - 1 > 0) ? y - 1 : 0; i < ((y + 2 < N) ? y + 2 : N); i++)
			for (int j = (x - 1 > 0) ? x - 1 : 0; j < ((x + 2 < N) ? x + 2 : N); j++)
			{
				totalNeighbours += data[((currentGen - 1) * N + i) * N + j];
			}
		//Attempt manual looping in cardinal directions to test speed
		totalNeighbours -= isAlive;
		bool currentStatus = (isAlive && !(totalNeighbours < 2 || totalNeighbours > 3)) || (totalNeighbours == 3);

		data[(currentGen * N + y) * N + x] = currentStatus;
		//data[((currentGen * N + y) * N + x) / warpSize] = currentStatus; //For bit packing indexing
	}
}

__global__ void CUDA_GL_gameLoop(bool* prevGen, bool* nextGen, uchar4* gl_buffer)
{
	//It is possible to place the 
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < N && x < N) {
		bool isAlive = prevGen[y * N + x];
		unsigned int totalNeighbours = 0;
		for (int i = (y - 1 > 0) ? y - 1 : 0; i < ((y + 2 < N) ? y + 2 : N); i++)
			for (int j = (x - 1 > 0) ? x - 1 : 0; j < ((x + 2 < N) ? x + 2 : N); j++)
			{
				totalNeighbours += prevGen[i * N + j];
			}
		//Attempt manual looping in cardinal directions to test speed
		totalNeighbours -= isAlive;
		bool currentStatus = (isAlive && !(totalNeighbours < 2 || totalNeighbours > 3)) || (totalNeighbours == 3);

		nextGen[y * N + x] = currentStatus;
		//printf("x, y: %d, %d | status: %d\n", x, y, currentStatus);
		gl_buffer[y * N + x].w = 255;
		gl_buffer[y * N + x].x = 255 * currentStatus;
		gl_buffer[y * N + x].y = 255 * currentStatus;
		gl_buffer[y * N + x].z = 255 * currentStatus;

		//gl_buffer[y * N + x].x = (x % 12 ==0) * 255;
		//gl_buffer[y * N + x].y = (x % 12 == 0) * 255;
		//gl_buffer[y * N + x].z = (x % 12 == 0) * 255;
	}
}

__global__ void BPKernel(uint32_t* data, int currentGen)
{
	//Only works if BLOCKSIZE is 32
	__shared__ bool packing[32];
	const uint16_t tid = threadIdx.x;

	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + tid;

	uint32_t cell = data[(((currentGen - 1) * N + y) * N + x) / BP];
	bool isAlive = (cell & (1 << (x % 32))) != 0;
	int totalNeighbours = 0;
	for (int i = (y - 1 > 0) ? y - 1 : 0; i < ((y + 2 < N) ? y + 2 : N); i++)
		for (int j = (x - 1 > 0) ? x - 1 : 0; j < ((x + 2 < N) ? x + 2 : N); j++)
		{
			uint32_t cell = data[(((currentGen - 1) * N + i) * N + j) / BP];
			totalNeighbours += (cell & (1 << (j % 32))) != 0;
		}
	totalNeighbours -= isAlive;
	bool currentStatus = (isAlive && !(totalNeighbours < 2 || totalNeighbours > 3)) || (totalNeighbours == 3);

	packing[tid] = currentStatus;

	//printf("x, y: %d, %d | status: %d\n", x, y, currentStatus);
	//HERE all threads in block sync to submit the mask to the final Buffer
	data[((currentGen * N + y) * N + x) / BP] = __ballot_sync(FULL_MASK, packing[tid]);
	//https://stackoverflow.com/questions/39488441/how-to-pack-bits-efficiently-in-cuda/39488714#39488714
	printf("x, y: % d, % d | status : %d, wasAlive: %d | total: %d\n", x, y, currentStatus, isAlive, data[((currentGen * N + y) * N + x) / BP]);
}


void printDevProp(cudaDeviceProp devProp)
{
	printf("Major revision number:         %d\n", devProp.major);
	printf("Minor revision number:         %d\n", devProp.minor);
	printf("Name:                          %s\n", devProp.name);
	printf("Total global memory:           %u\n", devProp.totalGlobalMem);
	printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %u\n", devProp.memPitch);
	printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n", devProp.clockRate);
	printf("Total constant memory:         %u\n", devProp.totalConstMem);
	printf("Texture alignment:             %u\n", devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return;
}

//void printDeviceMemory(void* deviceptr)
//{
//	size_t size;
//	cuMemGetAddressRange(NULL, size, deviceptr);
//}


void runKernel(int gen, bool* device_A)
{
	dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
	dim3 gridSize(N / BLOCKSIZE, N / BLOCKSIZE);
	//#pragma warning disable E0029
	CUDAgameLoop << <gridSize, blockSize >> > (device_A, gen);
}

void runBP(int gen, uint32_t* device_A) 
{
	dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
	dim3 gridSize(N / BLOCKSIZE, N / BLOCKSIZE);
	//#pragma warning disable E0029
	BPKernel << <gridSize, blockSize >> > (device_A, gen);
}


void render(cudaGraphicsResource* cuda_pbo_resource, bool* prevGen, bool* currGen)
{
	uchar4* d_out = 0;
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_out, NULL, cuda_pbo_resource);
	dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
	dim3 gridSize(N / BLOCKSIZE, N / BLOCKSIZE);
	CUDA_GL_gameLoop << <gridSize, blockSize >> > (prevGen, currGen, d_out);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

