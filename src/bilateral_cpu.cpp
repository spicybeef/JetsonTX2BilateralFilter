#include <stdint.h>
#include <stdio.h>
#include <math.h>

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
inline float distance(int x0, int y0, int x1, int y1)
{
    return static_cast<float>(sqrt( (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) ));
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
inline float gaussian(float x, float mu, float sigma)
{
    return static_cast<float>(exp(-((x - mu) * (x - mu))/(2 * sigma * sigma)) / (2 * M_PI * sigma * sigma));
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