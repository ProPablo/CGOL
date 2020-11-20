#include "Helpers.h"
#include <fstream> //Important to only include here not in header file due to that header file leaking those libs to main
#include <iostream>
#include <chrono>
//This file is here to implement ONCE the definition declared in .h hfile

std::chrono::time_point<std::chrono::high_resolution_clock> last;
#define MAX_FILE 10000
float Time(const char* msg)
{
	float result_time = 0.0f;
	if (msg != NULL)
	{
		auto now = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> elapsed_seconds = now - last;
		result_time = elapsed_seconds.count();
		printf("%s: %0.5f seconds\n", msg, elapsed_seconds.count());
	}
	last = std::chrono::high_resolution_clock::now();
	return result_time;
}

char error[] = { "Error encountered" };
char* load_file(char* file_name) //The cpp file should not conatin default param
{
    //FILE* fp = fopen(file_name, "rb");
    //if (!fp)
    //{
    //    // print some error or throw exception here
    //    return error;
    //}
    //char* source = new char[max_size];
    //size_t source_size = fread(source, 1, max_size, fp);
    //fclose(fp);
    //if (!source_size)
    //{
    //    delete[] source;
    //    // print some error or throw exception here
    //    return error;
    //}
    //return source;

	std::ifstream infile{ file_name };
	std::string program_source{ std::istreambuf_iterator<char>(infile), std::istreambuf_iterator<char>() };
	char src[MAX_FILE];
	program_source.copy(src, program_source.size() + 1);
	src[program_source.size()] = '\0';
	return src;
}

void matSerealizer(bool* data)
{
	std::ofstream out("cereal", std::ios::out);
	for (int gen = 0; gen < generations; gen++)
	{
		for (int x = 0; x < N; x++)
		{
			for (int y = 0; y < N; y++)
			{
				if (data[(gen * N + y) * N + x] == true)
				{
					out << "0";
				}
				else
				{
					out << ".";
				}
			}
			out << "\n";
		}
		out << "----STARTING NEW GENERATION----\n";
	}
	out.close();
}

bool* BPConverter(uint32_t* data) 
{

	bool* newBools = (bool*)malloc(sizeOfArray);
	for (int gen = 0; gen < generations; gen++)
	{
		for (int x = 0; x < N; x++)
		{
			for (int y = 0; y < N; y++)
			{
				uint32_t cell = data[((gen*N + y) * N + x )/ BP];
				bool alive = (cell & (1 << (x % 32))) != 0;
				newBools[(gen * N + y) * N + x] = alive;
			}
		}
	
	}
	return newBools;
}

void printArray(int* arr, int size)
{
	for (int i = 0; i < size; i++)
		std::cout << arr[i] << ", ";
	std::cout << '\n';
}

void cleanMat(bool* mat)
{
	for(int gen = 0; gen<generations; gen++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
			{
				mat[(gen * N + i) * N + j] = 0;
			}
}

void cleanBP(uint32_t* mat) 
{
	for (int gen = 0; gen < generations; gen++)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
			{
				mat[((gen * N + i) * N + j)/ 32] = 0;
			}
}