
#include <iostream>

#include <iomanip>
#include <stdio.h>

#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

#include "Helpers.h"
#include "perlin/Perlin.h"
#include "gol/gol.h"

#define TIMELINES 1
#define SIZES 5
#define SIZE_INCREMENT 500

class CGOL_Draw : public olc::PixelGameEngine
{
private:
	bool* data;
	int gen = 0;
	float elapsedTime = 0;

public:
	CGOL_Draw(bool* Data)
	{
		data = Data;
		sAppName = "Conway's Game of Life";
	}

public:
	bool OnUserCreate() override
	{
		// Called once at the start, so create things here
		printf("size: %d, %d\n", ScreenWidth(), ScreenHeight());
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		// called once per frame
		if (gen >= generations)
			return true;

		elapsedTime += fElapsedTime;
		if (elapsedTime < frameTime)
			return true;
		elapsedTime = 0;

		Clear(olc::DARK_BLUE);

		// called once per frame
		for (int x = 0; x < N; x++)
			for (int y = 0; y < N; y++)
				if (data[(gen * N + y) * N + x] == true)
					Draw(x, y, olc::Pixel(255, 255, 255));
		gen++;
		return true;
	}
};

void gameLoop(bool* data, int currentGen)
{
	for (int y = 0; y < N; y++)
	{
		for (int x = 0; x < N; x++)
		{
			//Treat generations (time) as the third dimension to array
			bool isAlive = data[((currentGen - 1) * N + y) * N + x];
			//An important note is to store repretetive access in local stack variable, compiler will place in register and avoids repeat memory access
			int totalNeighbours = 0;
			//Non modulous wraparound version of GOL due to parallel version
			for (int i = (y - 1 > 0) ? y - 1 : 0; i < ((y + 2 < N) ? y + 2 : N); i++)
				for (int j = (x - 1 > 0) ? x - 1 : 0; j < ((x + 2 < N) ? x + 2 : N); j++)
				{
					totalNeighbours += data[((currentGen - 1) * N + i) * N + j];
				}

			/*for (int i = y-1; i<y+2; i++)
				for (int j = x - 1; j < x + 2; j++) {
					if (i < 0 || i >= N || j < 0 || j >= N) continue;
					totalNeighbours += data[((currentGen - 1) * N + i) * N + j];
				}*/

				//Do not count self
			totalNeighbours -= isAlive;
			bool currentStatus = (isAlive && !(totalNeighbours < 2 || totalNeighbours > 3)) || (totalNeighbours == 3);
			data[(currentGen * N + y) * N + x] = currentStatus;
		}
	}
}

void BPgameLoop(uint32_t* data, int currentGen)
{
	for (int y = 0; y < N; y++)
	{
		for (int x = 0; x < N; x++)
		{
			uint32_t cell = data[(((currentGen - 1) * N + y) * N + x )/ BP];
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

			//This is fine to do because array is initialised to 0 (not reusing array)
			data[((currentGen * N + y) * N + x )/ BP] |= (currentStatus << (x % 32));
		}
	}
}


int mainTEST(int argc, char* argv[])
{

	bool* host_A = (bool*)malloc(sizeOfArray);
	if (host_A == NULL)
	{
		printf("memalloc failure\n");
		return EXIT_FAILURE;
	}

	//Code for running tests
	/*for (int size = 0; size < SIZES; size++) {

	}*/
	//std::ofstream out("parallelresults.csv", std::ios::app);
	//out << "OC: " << " No bit packing, size: " << N << '\n';
	//for (int timeline = 0; timeline < TIMELINES; timeline++)
	//{
	int gen = 0;
	cleanMat(host_A);
	//bool* device_A = initCUDA(host_A);
	bool* device_A = initCUDA<bool>(host_A, sizeOfArray);
	Time(NULL);

	CUDAinitPerlin(SEED, N);
	generatePerlin(device_A);
	//initPerlin(SEED);
	//for (int i = 0; i < N; i++)
	//	for (int j = 0; j < N; j++)
	//	{
	//		host_A[(gen * N + i) * N + j] = noise((double)j * OFFSET_SCALE, (double)i * OFFSET_SCALE, 0) <= INCLINITATION;
	//	}

	gen++;
	Time("Finished generating Perlin");

	

#ifdef PARALLEL
	Time(NULL);
	for (; gen < generations; gen++)
	{
		runKernel(gen, device_A);
	}
	endCUDA<bool>(host_A, device_A);
	//out << timeline << "," << Time("Finishing time for parallel") << '\n';
#else
	Time(NULL);
	for (; gen < generations; gen++)
	{
		gameLoop(host_A, gen);
	}
	out << timeline << "," << Time("Finishing time for serial") << '\n';
#endif
	//}
	//out.close();

	//matSerealizer(host_A);

	CGOL_Draw demo(host_A);
	if (demo.Construct(N, N, pixelSize, pixelSize, false, true))
		demo.Start();
	int input;
	std::cout << "BRUH\n";
	std::cin >> input;
	return 0;
}



int mainBP(int argc, char* argv[])
{
	uint32_t* host_A = (uint32_t*)malloc(sizeOfBP);
	if (host_A == NULL)
	{
		printf("memalloc failure\n");
		return EXIT_FAILURE;
	}

	int gen = 0;
	cleanBP(host_A);
	//bool* device_A = initCUDA(host_A);
	Time(NULL);

	initPerlin(SEED);
	//Continue using sequential perlin for sake of demo
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			//Anywhere the actual indexing is required, i or j remain the same, only when indexing main array
			host_A[((gen * N + i) * N + j) / 32] |= ((noise((double)j * OFFSET_SCALE, (double)i * OFFSET_SCALE, 0) <= INCLINITATION) << (j%32));
		}
	gen++;

#ifdef PARALLEL

	uint32_t* device_A = initCUDA<uint32_t>(host_A, sizeOfBP);
	std::cout << "Starting parallel";
	for (; gen < generations; gen++)
	{
		runBP(gen, device_A);
	}
	std::cout << "Ended parallel";
	endCUDA<uint32_t>(host_A, device_A);
	//out << timeline << "," << Time("Finishing time for parallel") << '\n';
#else
	for (; gen < generations; gen++)
	{
		BPgameLoop(host_A, gen);
	}
#endif

	bool* converted = BPConverter(host_A);
	matSerealizer(converted);

	CGOL_Draw demo(converted);
	if (demo.Construct(N, N, pixelSize, pixelSize, false, true))
		demo.Start();
}