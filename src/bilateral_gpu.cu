#include <stdint.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "bilateral_gpu.h"

const int BLOCKDIM = 16;

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
__device__ inline float distance(int x0, int y0, int x1, int y1)
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
__device__ inline float gaussian(float x, float mu, float sigma)
{
    return static_cast<float>(expf(-((x - mu) * (x - mu))/(2 * sigma * sigma)) / (2 * M_PI * sigma * sigma));
}

/**
 * @brief      A naive implementation of the bilateral filter
 *
 * @param      inputImage   The input float array
 * @param      outputImage  The output float array
 * @param[in]  rows         The number of rows in the image
 * @param[in]  cols         The number of columns in the image
 * @param[in]  window       The window to use in the filter
 * @param[in]  sigmaD       The distance parameter
 * @param[in]  sigmaR       The intensity parameter
 */
__global__ void bilateralNaiveGpuKernel(
    float* inputImage,
    float* outputImage,
    int rows, int cols,
    uint32_t window,
    float sigmaD,
    float sigmaR)
{
    float filteredPixel, neighbourPixel, currentPixel;
    float wP, gR, gD;
    int neighbourCol;
    int neighbourRow;

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= cols || row >= rows)
    {
        return;
    }

    filteredPixel = 0;
    wP = 0;

    for (int windowCol = 0; windowCol < window; windowCol++)
    {
        for (int windowRow = 0; windowRow < window; windowRow++)
        {
            neighbourCol = col - (window / 2) - windowCol;
            neighbourRow = row - (window / 2) - windowRow;

            // Prevent us indexing into regions that don't exist
            if (neighbourCol < 0)
            {
                neighbourCol = 0;
            }
            if (neighbourRow < 0)
            {
                neighbourRow = 0;
            }

            neighbourPixel = inputImage[neighbourCol + neighbourRow * cols];
            currentPixel = inputImage[col + row * cols];

            // Intensity factor
            gR = gaussian(neighbourPixel - currentPixel, 0.0, sigmaR);
            // Distance factor
            gD = gaussian(distance(col, row, neighbourCol, neighbourRow), 0.0, sigmaD);

            filteredPixel += neighbourPixel * (gR * gD);

            wP += (gR * gD);
        }
    }
    outputImage[col + row * cols] = filteredPixel / wP;
}

__global__ void bilateralOptimizedGpuColsKernel(
    float* inputImage,
    float* outputImage,
    int rows, int cols,
    uint32_t window,
    float sigmaD,
    float sigmaR)
{
    float filteredPixel, neighbourPixel, currentPixel;
    float wP, gR, gD;
    int neighbourCol;

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= cols || row >= rows)
    {
        return;
    }

    filteredPixel = 0;
    wP = 0;

    #pragma unroll
    for (int windowCol = 0; windowCol < window; windowCol++)
    {
        neighbourCol = col - (window / 2) - windowCol;

        // Prevent us indexing into regions that don't exist
        if (neighbourCol < 0)
        {
            neighbourCol = 0;
        }

        neighbourPixel = inputImage[neighbourCol + row * cols];
        currentPixel = inputImage[col + row * cols];

        // Intensity factor
        gR = gaussian(neighbourPixel - currentPixel, 0.0, sigmaR);
        // Distance factor
        gD = gaussian(distance(col, row, neighbourCol, row), 0.0, sigmaD);

        filteredPixel += neighbourPixel * (gR * gD);

        wP += (gR * gD);
    }
    outputImage[col + row * cols] = filteredPixel / wP;
}

__global__ void bilateralOptimizedGpuRowsKernel(
    float* inputImage,
    float* outputImage,
    int rows, int cols,
    uint32_t window,
    float sigmaD,
    float sigmaR)
{
    float filteredPixel, neighbourPixel, currentPixel;
    float wP, gR, gD;
    int neighbourRow;

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= cols || row >= rows)
    {
        return;
    }

    filteredPixel = 0;
    wP = 0;

    #pragma unroll
    for (int windowRow = 0; windowRow < window; windowRow++)
    {
        neighbourRow = row - (window / 2) - windowRow;

        // Prevent us indexing into regions that don't exist
        if (neighbourRow < 0)
        {
            neighbourRow = 0;
        }

        neighbourPixel = inputImage[col + neighbourRow * cols];
        currentPixel = inputImage[col + row * cols];

        // Intensity factor
        gR = gaussian(neighbourPixel - currentPixel, 0.0, sigmaR);
        // Distance factor
        gD = gaussian(distance(col, row, col, neighbourRow), 0.0, sigmaD);

        filteredPixel += neighbourPixel * (gR * gD);

        wP += (gR * gD);
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
    
    cudaStatus = cudaMalloc<float>(&gpuInput, rows * cols * sizeof(float));
    checkCudaErrors(cudaStatus);    
    cudaStatus = cudaMalloc<float>(&gpuOutput, rows * cols * sizeof(float));
    checkCudaErrors(cudaStatus);

    cudaStatus = cudaMemcpy(gpuInput, inputImage, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaStatus);    
    
    const dim3 block(BLOCKDIM, BLOCKDIM);
    const dim3 grid((cols / BLOCKDIM) + (cols % BLOCKDIM), (rows / BLOCKDIM) + (rows % BLOCKDIM));

    // printf("BlockDimensions x=%d, y=%d, z=%d\n", block.x, block.y, block.z);
    // printf("GridDimensions x=%d, y=%d, z=%d\n", grid.x, grid.y, grid.z);

    bilateralNaiveGpuKernel<<<grid,block>>>(gpuInput, gpuOutput, rows, cols, window, sigmaD, sigmaR);

    cudaStatus = cudaDeviceSynchronize();
    checkCudaErrors(cudaStatus);    

    cudaStatus = cudaMemcpy(outputImage, gpuOutput, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaStatus);    

    cudaStatus = cudaFree(gpuInput);
    checkCudaErrors(cudaStatus);    
    cudaStatus = cudaFree(gpuOutput);
    checkCudaErrors(cudaStatus);    
}

void bilateralOptimizedGpu(
    float* inputImage,
    float* outputImage,
    int rows, int cols,
    uint32_t window,
    float sigmaD,
    float sigmaR)
{
    float* gpuInput;
    float* gpuBuffer;
    float* gpuOutput;
    cudaError_t cudaStatus; 
    
    cudaStatus = cudaMalloc<float>(&gpuInput, rows * cols * sizeof(float));
    checkCudaErrors(cudaStatus);    
    cudaStatus = cudaMalloc<float>(&gpuBuffer, rows * cols * sizeof(float));
    checkCudaErrors(cudaStatus);
    cudaStatus = cudaMalloc<float>(&gpuOutput, rows * cols * sizeof(float));
    checkCudaErrors(cudaStatus);

    cudaStatus = cudaMemcpy(gpuInput, inputImage, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaStatus);    
    
    const dim3 block(BLOCKDIM, BLOCKDIM);
    const dim3 grid((cols / BLOCKDIM) + (cols % BLOCKDIM), (rows / BLOCKDIM) + (rows % BLOCKDIM));

    bilateralOptimizedGpuColsKernel<<<grid,block>>>(gpuInput, gpuBuffer, rows, cols, window, sigmaD, sigmaR);
    bilateralOptimizedGpuRowsKernel<<<grid,block>>>(gpuBuffer, gpuOutput, rows, cols, window, sigmaD, sigmaR);

    cudaStatus = cudaDeviceSynchronize();
    checkCudaErrors(cudaStatus);    

    cudaStatus = cudaMemcpy(outputImage, gpuOutput, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaStatus);    

    cudaStatus = cudaFree(gpuInput);
    checkCudaErrors(cudaStatus);
    cudaStatus = cudaFree(gpuBuffer);
    checkCudaErrors(cudaStatus);    
    cudaStatus = cudaFree(gpuOutput);
    checkCudaErrors(cudaStatus);    
}