#include <stdlib.h>
#include <iostream>
#include <sstream> 
#define GLEW_STATIC
//Now make sure to link the static file instead of the normal lib file
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include "Helpers.h"
#include "perlin/Perlin.h"
#include "gol/gol.h"


GLFWwindow* window;
GLuint pbo = 0;
GLuint tex = 0;
cudaGraphicsResource* cuda_pbo_resource;

cudaError_t cudaStatus;

bool* device_A;

double lastTime;
double fLasttime;
int nbFrames;
double elapsedTime;

const int sizeOfCUDABuffer = sizeofSingle * 2;

void initGL()
{
	/* Initialize the library */
	if (!glfwInit())
		exit(-1);
	///* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(640, 480, PROGRAM_NAME, NULL, NULL);
	//window = glfwCreateWindow(N, N, PROGRAM_NAME, NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		exit(-1);
	}
	///* Make the window's context current */
	glfwMakeContextCurrent(window);

	glewInit();
}

void initGLBuffer()
{
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * N * N * sizeof(GLubyte), 0, GL_DYNAMIC_DRAW);
	/*glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);*/
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}


void drawTexture() {
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, N, N, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	//glEnable(GL_TEXTURE_2D);
	//glBegin(GL_QUADS);
	//glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	//glTexCoord2f(0.0f, 1.0f); glVertex2f(0, N);
	//glTexCoord2f(1.0f, 1.0f); glVertex2f(N, N);
	//glTexCoord2f(1.0f, 0.0f); glVertex2f(N, 0);
	//glEnd();
	//glDisable(GL_TEXTURE_2D);

	glDrawPixels(N, N, GL_RGBA, GL_UNSIGNED_BYTE, 0);
}

void initCUDABuffer(bool* host_A) {
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
		cudaFree(device_A);
		exit(cudaStatus);
	}

	cudaStatus = cudaMalloc((void**)&device_A, sizeOfCUDABuffer);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaMalloc failed!";
		cudaFree(device_A);
		exit(cudaStatus);
	}

	cudaStatus = cudaMemcpy(device_A, host_A, sizeOfCUDABuffer, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaMemcpy failed!";
		cudaFree(device_A);
		exit(cudaStatus);
	}
}

void endCUDAGL()
{
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
	cudaFree(device_A);
}

int showFPS()
{
	// Measure speed
	double currentTime = glfwGetTime();
	double delta = currentTime - lastTime; //inseconds
	//This is the FPS LIMITER
#ifdef FPS_CAP
	double fElapsedTime = currentTime - fLasttime;
	elapsedTime += fElapsedTime;
	fLasttime = currentTime;
	if (elapsedTime < frameTime) {
		return 0;
	}
	elapsedTime = 0;
#endif
	nbFrames++;
	if (delta >= 1.0) { // If last cout was more than 1 sec ago
		//std::cout << 1000.0 / double(nbFrames) << '\n';

		double fps = double(nbFrames) / delta;

		std::stringstream ss;
		ss << PROGRAM_NAME << " [" << fps << " FPS]";

		glfwSetWindowTitle(window, ss.str().c_str());

		nbFrames = 0;
		lastTime = currentTime;
	}
	return 1;
}


void GL_gameLoop(bool* prevGen, bool* nextGen, uchar4* gl_buffer)
{
	for (int y = 0; y < N; y++)
	{
		for (int x = 0; x < N; x++) {
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
			gl_buffer[y * N + x].w = 255;
			gl_buffer[y * N + x].x = 255 * currentStatus;
			gl_buffer[y * N + x].y = 255 * currentStatus;
			gl_buffer[y * N + x].z = 255 * currentStatus;
		}
	}
}



void seqRender() {

}
#ifdef OpenGL_DRAW
int main(int argc, char* argv[])
{
	long gen = 0;
	bool* A = (bool*)malloc(N * N * 2);
	//bool* B = (bool*)malloc(N * N);
	if (A == NULL)
		return EXIT_FAILURE;

	initPerlin(SEED);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			A[(gen * N + i) * N + j] = noise((double)j * OFFSET_SCALE, (double)i * OFFSET_SCALE, 0) <= INCLINITATION;
		}
	gen++;

	initGL();
	initCUDABuffer(A);
	initGLBuffer();

	///* Loop until the user closes the window */
	while (!glfwWindowShouldClose(window))
	{
		if (!showFPS()) {
			//std::cout << "skipping";
			continue;
		}
		bool isOdd = !(gen % 2);
		//std::cout << "Is odd: " << sizeofSingle * (!isOdd) << "Generation: " << gen << '\n';

		/* Render here */
		glClear(GL_COLOR_BUFFER_BIT);
		//If even (0) then B1 will be previous and B2 will be next, 
#ifdef PARALLEL
		render(cuda_pbo_resource, &device_A[sizeofSingle * isOdd], &device_A[sizeofSingle * (!isOdd)]);
#else
		seqRender();
#endif // PARALLEL


		drawTexture();
		//	/* Swap front and back buffers */
		glfwSwapBuffers(window);

		//TODO: handle fps

		//	/* Poll for and process events */
		glfwPollEvents();
		gen++;
	}

	glfwTerminate();
	//TODO: delete hostMEm
}
#endif