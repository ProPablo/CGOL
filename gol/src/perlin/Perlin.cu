//#include "Perlin.h"
#include "../Helpers.h"
#include "cuda_runtime.h"
#include <cmath>
#include <algorithm> // std::shuffle
#include <random>	 // std::default_random_engine
#include <array>

__constant__ int perlin_A[512];

//__device__ __constant__ int N_size_P;

int _N_size_P;

//Way too much work for unnecessary parellization of just shuffling an array
void CUDAinitPerlin(int seed, int size)
{
	int permutation[] = { 151, 160, 137, 91, 90, 15,
						 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23,
						 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
						 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
						 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
						 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196,
						 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123,
						 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
						 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
						 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
						 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107,
						 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
						 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180 };
	//shuffle Needs end +1
	if (seed != -1)
	{
		std::shuffle(&permutation[0], &permutation[256], std::default_random_engine(seed));
	}
	int p[512];
	for (int i = 0; i < 256; i++)
		p[256 + i] = p[i] = permutation[i];

	cudaMemcpyToSymbol(perlin_A, &p, 512 * sizeof(int));
	//cudaMemcpyToSymbol(N_size_P, &size, sizeof(int));
	_N_size_P = size;
}

__device__ double _fade(double t) { return t * t * t * (t * (t * 6 - 15) + 10); }
__device__ double _lerp(double t, double a, double b) { return a + t * (b - a); }
__device__ double _grad(int hash, double x, double y, double z)
{
	int h = hash & 15;		  // CONVERT LO 4 BITS OF HASH CODE
	double u = h < 8 ? x : y, // INTO 12 GRADIENT DIRECTIONS.
		v = h < 4 ? y : h == 12 || h == 14 ? x : z;
	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}


__device__ double CUDAnoise(double x, double y, double z)
{
	int X = (int)floor(x) & 255, // FIND UNIT CUBE THAT
		Y = (int)floor(y) & 255, // CONTAINS POINT.
		Z = (int)floor(z) & 255;
	x -= floor(x); // FIND RELATIVE X,Y,Z
	y -= floor(y); // OF POINT IN CUBE.
	z -= floor(z);

	double u = _fade(x), // COMPUTE FADE CURVES
		v = _fade(y),	// FOR EACH OF X,Y,Z.
		w = _fade(z);

	int A = perlin_A[X] + Y, AA = perlin_A[A] + Z, AB = perlin_A[A + 1] + Z,		// HASH COORDINATES OF
		B = perlin_A[X + 1] + Y, BA = perlin_A[B] + Z, BB = perlin_A[B + 1] + Z; // THE 8 CUBE CORNERS,

	double res = _lerp(w, _lerp(v, _lerp(u, _grad(perlin_A[AA], x, y, z), _grad(perlin_A[BA], x - 1, y, z)), _lerp(u, _grad(perlin_A[AB], x, y - 1, z), _grad(perlin_A[BB], x - 1, y - 1, z))), _lerp(v, _lerp(u, _grad(perlin_A[AA + 1], x, y, z - 1), _grad(perlin_A[BA + 1], x - 1, y, z - 1)), _lerp(u, _grad(perlin_A[AB + 1], x, y - 1, z - 1), _grad(perlin_A[BB + 1], x - 1, y - 1, z - 1))));
	return (res + 1.0) / 2.0;
}

__global__ void perlinKernel(bool* data, int N_size_P)
{
	//Indexing similiar to 2d flattening
	//Treating blocks as 2nd dim (y), and thread as 1st dim (x)
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	data[y * N_size_P + x] = CUDAnoise((double)x * OFFSET_SCALE, (double)y * OFFSET_SCALE, 0) <= INCLINITATION;
}
void generatePerlin(bool* data)
{
	dim3 gridSize(_N_size_P/ BLOCKSIZE, _N_size_P / BLOCKSIZE);
	dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
	perlinKernel << <gridSize, blockSize >> > (data, _N_size_P);
}