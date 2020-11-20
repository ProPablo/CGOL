#pragma once

//Extern means the array must be initialised somewhere else kind of like the functions
//Use inline intead to create editable

//#ifndef __OPENCL_C_VERSION__
//	#include <cmath>
//#endif
double noise(double x, double y, double z);
void initPerlin(int seed);

void CUDAinitPerlin(int seed, int size);
void generatePerlin(bool* data);
