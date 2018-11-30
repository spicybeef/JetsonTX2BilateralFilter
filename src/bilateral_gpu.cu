#include <stdint.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "bilateral_gpu.h"

const int BLOCKDIM = 32;

/**
 * @brief      Calculates the Euclidean distance between two points (x0, y0) and
 *             (x1, y1)
 *
 * @param[in]  x0    The x0 coordinate
 * @param[in]  y0    The y0 coordinate
 * @param[in]  x1    The x1 coordinate
 * @param[in]  y1    The y1 coordinate
 *
 * @return     The distance between the two points
 */
__device__ float distance(int x0, int y0, int x1, int y1)
{
    return static_cast<float>(sqrtf( (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) ));
}

/**
 * @brief      Calculates the one-dimensional Gaussian function for a given
 *             point x
 *
 * @param[in]  x      The point on the distribution to return a value from
 * @param[in]  mu     The mean value
 * @param[in]  sigma  The standard deviation of the distribution
 *
 * @return     The value of the 1D Gaussian function at point x with mean mu and
 *             standard deviation sigma
 */
__device__ float gaussian(float x, float mu, float sigma)
{
    return static_cast<float>(exp(-((x - mu) * (x - mu))/(2 * sigma * sigma)) / (2 * M_PI * sigma * sigma));
}

__global__ void bilateralGpuKernel(
    float* inputImage,
    float* outputImage,
    int rows, int cols,
    uint32_t window,
    float sigmaD,
    float sigmaR)
{
    float filteredPixel;
    float wP, gR, gD;
    int neighborCol;
    int neighborRow;

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    filteredPixel = 0;
    wP = 0;

    for (int windowCol = 0; windowCol < window; windowCol++)
    {
        for (int windowRow = 0; windowRow < window; windowRow++)
        {
            neighborCol = col - (window / 2) - windowCol;
            neighborRow = row - (window / 2) - windowRow;

            // Prevent us indexing into regions that don't exist
            if (neighborCol < 0)
            {
                neighborCol = 0;
            }
            if (neighborRow < 0)
            {
                neighborRow = 0;
            }

            // Intensity factor
            gR = gaussian(inputImage[neighborCol + neighborRow * cols] - inputImage[col + row * cols], 0.0, sigmaR);
            // Distance factor
            gD = gaussian(distance(col, row, neighborCol, neighborRow), 0.0, sigmaD);

            filteredPixel += inputImage[neighborCol + neighborRow * cols] * (gR * gD);

            wP += (gR * gD);
        }
    }
    outputImage[col + row * cols] = filteredPixel / wP;
}

void bilateralNaiveGpu(
    float* inputImage,
    float* outputImage,
    int rows, int cols,
    uint32_t window,
    float sigmaD,
    float sigmaR)
{
    float* gpuInput;
    float* gpuOutput;
    cudaError_t cudaStatus; 
    
    cudaStatus = cudaMalloc<float>(&gpuInput, rows * cols);
    checkCudaErrors(cudaStatus);    
    cudaStatus = cudaMalloc<float>(&gpuOutput, rows * cols);
    checkCudaErrors(cudaStatus);

    cudaStatus = cudaMemcpy(gpuInput, inputImage, rows * cols, cudaMemcpyHostToDevice);
    checkCudaErrors(cudaStatus);    
    
    const dim3 block(BLOCKDIM, BLOCKDIM);
    const dim3 grid(cols / BLOCKDIM, rows / BLOCKDIM);

    bilateralGpuKernel<<<grid,block>>>(gpuInput, gpuOutput, rows, cols, window, sigmaD, sigmaR);

    cudaStatus = cudaDeviceSynchronize();
    checkCudaErrors(cudaStatus);    

    cudaStatus = cudaMemcpy(outputImage, gpuOutput, rows * cols,cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaStatus);    

    cudaStatus = cudaFree(gpuInput);
    checkCudaErrors(cudaStatus);    
    cudaStatus = cudaFree(gpuOutput);
    checkCudaErrors(cudaStatus);    
}
