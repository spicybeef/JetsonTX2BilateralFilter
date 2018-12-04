#include <stdint.h>
#include <stdio.h>
#include <math.h>

extern "C"
{
    // This is defined in main.cpp
    float expf_michel (float x);
}

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
float distance(int x0, int y0, int x1, int y1)
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
float gaussian(float x, float mu, float sigma)
{
    return static_cast<float>(expf_michel(-((x - mu) * (x - mu))/(2 * sigma * sigma)) / (2 * M_PI * sigma * sigma));
}

/**
 * @brief      A (very) naive implementation of the bilateral filter
 *
 * @param      inputFloat   The input float array
 * @param      outputFloat  The output float array
 * @param[in]  rows         The number of rows in the image
 * @param[in]  cols         The number of columns in the image
 * @param[in]  window       The window to use in the filter
 * @param[in]  sigmaD       The distance parameter
 * @param[in]  sigmaR       The intensity parameter
 */
void bilateralNaiveCpu(float* inputFloat, float* outputFloat, int rows, int cols, uint32_t window, float sigmaD, float sigmaR)
{
    float filteredPixel, neighbourPixel, currentPixel;
    float wP, gR, gD;
    int neighbourCol;
    int neighbourRow;

    for (int col = 0; col < cols; col++)
    {
        for (int row = 0; row < rows; row++)
        {
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

                    neighbourPixel = inputFloat[neighbourCol + neighbourRow * cols];
                    currentPixel = inputFloat[col + row * cols];

                    // Intensity factor
                    gR = gaussian(neighbourPixel - currentPixel, 0.0, sigmaR);
                    // Distance factor
                    gD = gaussian(distance(col, row, neighbourCol, neighbourRow), 0.0, sigmaD);

                    filteredPixel += neighbourPixel * (gR * gD);

                    wP += (gR * gD);
                }
            }
            outputFloat[col + row * cols] = filteredPixel / wP;
        }
    }
}

/**
 * @brief      An optimized version of the CPU bilateral filter. Optmizations
 *             include using OpenMP for threading of the main loop and changing
 *             memory access to be sequential by turning the outer loops into a
 *             single loop.
 *
 * @param      inputFloat   The input float array
 * @param      outputFloat  The output float array
 * @param[in]  rows         The number of rows in the image
 * @param[in]  cols         The number of columns in the image
 * @param[in]  window       The window to use in the filter
 * @param[in]  sigmaD       The distance parameter
 * @param[in]  sigmaR       The intensity parameter
 */
void bilateralOptimizedCpu(float* inputFloat, float* outputFloat, int rows, int cols, uint32_t window, float sigmaD, float sigmaR)
{
    int totalPixels = rows * cols;
    float* buffer = new float [rows * cols];

    // Moved variable declaration into for loops as each index now becomes its
    // own thread. Using OpenMP pragmas to enable multithreading of individual
    // pixel computations. Split into column and row convolutions.
    #pragma omp parallel for
    for (int pixel = 0; pixel < totalPixels; pixel++)
    {
        float filteredPixel = 0;
        float wP = 0;
        int row = pixel / cols;
        int col = pixel % cols;
        for (int windowCol = 0; windowCol < window; windowCol++)
        {
            int neighbourCol = col - (window / 2) - windowCol;

            // Prevent us indexing into regions that don't exist
            if (neighbourCol < 0)
            {
                neighbourCol = 0;
            }
            
            float neighbourPixel = inputFloat[neighbourCol + row * cols];
            float currentPixel = inputFloat[pixel];

            // Intensity factor
            float gR = gaussian(neighbourPixel - currentPixel, 0.0, sigmaR);
            // Distance factor
            float gD = gaussian(distance(col, row, neighbourCol, row), 0.0, sigmaD);

            filteredPixel += neighbourPixel * (gR * gD);

            wP += (gR * gD);
        }
        buffer[pixel] = filteredPixel / wP;
    }

    #pragma omp parallel for
    for (int pixel = 0; pixel < totalPixels; pixel++)
    {
        float filteredPixel = 0;
        float wP = 0;
        int row = pixel / cols;
        int col = pixel % cols;
        for (int windowRow = 0; windowRow < window; windowRow++)
        {
            int neighbourRow = row - (window / 2) - windowRow;

            // Prevent us indexing into regions that don't exist
            if (neighbourRow < 0)
            {
                neighbourRow = 0;
            }
            
            float neighbourPixel = buffer[col + neighbourRow * cols];
            float currentPixel = buffer[pixel];

            // Intensity factor
            float gR = gaussian(neighbourPixel - currentPixel, 0.0, sigmaR);
            // Distance factor
            float gD = gaussian(distance(col, row, col, neighbourRow), 0.0, sigmaD);

            filteredPixel += neighbourPixel * (gR * gD);

            wP += (gR * gD);
        }
        outputFloat[pixel] = filteredPixel / wP;
    }
}