 #include <cstdio>
 #include <cstdlib>
#include <cctype>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
 #include <helper_cuda.h>
#include <cuda.h>

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}


int
main(void)
{

	cudaError_t err = cudaSuccess;

	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	printf("[%d elemanlý vektor ekleme]\n", numElements);

	float *h_A = (float *)malloc(size);

	float *h_B = (float *)malloc(size);

	float *h_C = (float *)malloc(size);


	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, " Ana vektorleri ayirma hatasi!\n");
		exit(EXIT_FAILURE);
	}


	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}


	float *d_A = NULL;
	err = cudaMalloc((void **)&d_A, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, " A device vektorunu ayirma hatasi (hata kodu %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_B = NULL;
	err = cudaMalloc((void **)&d_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "B device vektorunu ayirma hatasi(hata kodu %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_C = NULL;
	err = cudaMalloc((void **)&d_C, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "C device vektorunu ayirma hatasi (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Giriþ verilerini ana bellekten CUDA cihazýna kopyalayýn\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "A vektoru host tan device a kopyalanamadi(hata kodu %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "B vektoru host tan device a kopyalanamadi (hata kodu %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, numElements);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "VectorAdd çekirdeði baþlatýlamadý(hata kodu %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	printf("Çýktý verilerini CUDA cihazýndan ana bilgisayar belleðine kopyalayýn\n");
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "C vektoru device tan host a kopyalanamadi (hata kodu %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	for (int i = 0; i < numElements; ++i)
	{
		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
		{
			fprintf(stderr, "Ogede sonuc dogrulamasi basarisiz oldu %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}

	printf("Test PASSED\n");


	err = cudaFree(d_A);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "A vektoru bosaltilamadi (hata kodu %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_B);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "B vektoru bosaltilamadi (hata kodu %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_C);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "C vektoru bosaltilamadi (hata kodu %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	free(h_A);
	free(h_B);
	free(h_C);

	printf("Bitti\n");
	return 0;
}

