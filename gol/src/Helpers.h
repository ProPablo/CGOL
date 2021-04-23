#pragma once //#ifndef HELPER_H    pragma once replaces this directive

#include <stdint.h>

#define PROGRAM_NAME "CGOL"

#define generations 3
//Has to be a multiple of 32
#define N 320

#define pixelSize 2
#define frameTime 0.0005f
#define frameTimeMilli (frameTime * 1000) 

#define SEED 42

//This controls how much variance, higher means more
#define OFFSET_SCALE 0.1

#define INCLINITATION 0.3

//Parallel or sequential directive
#define PARALLEL
#define BLOCKSIZE 32

#define BP 32

//#define P_PERLIN

//#define OpenGL_DRAW
//#define FPS_CAP

const int sizeofSingle = N * N * sizeof(bool);
const int sizeOfArray = generations * sizeofSingle;

const int sizeOfBP = (N * N * generations / 32) * sizeof(uint32_t);

float Time(const char* msg);
char* load_file(char* file_name, int max_size = 0x100000);
void matSerealizer(bool* data);
void printArray(int* arr, int size);
void cleanMat(bool* mat);
void cleanBP(uint32_t* mat);
bool* BPConverter(uint32_t* data);