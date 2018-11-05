#include <stdint.h>
#include <cmath>

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
float gaussian(float x, float mu, float sigma)
{
    return static_cast<float>(exp(-((x - mu) * (x - mu))/(2 * sigma * sigma)) / (2 * M_PI * sigma * sigma));
}

/**
 * @brief      A (very) naive implementation of the bilateral filter
 *
 * @param      inputFloat   The input float array
 * @param      outputFloat  The output float array
 * @param[in]  rows         The number of rows in the image
 * @param[in]  cols         The number of columsn in the image
 * @param[in]  window       The window to use in the filter
 * @param[in]  sigmaD       The distance parameter
 * @param[in]  sigmaR       The intensity parameter
 */
void bilateralNaive(float* inputFloat, float** outputFloat, int rows, int cols, uint32_t window, float sigmaD, float sigmaR)
{
    float filteredPixel;
    float wP, gR, gD;
    int neighborCol;
    int neighborRow;

    // Instantiate new output float array
    float* output;
    output = new float [rows * cols];

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
                    neighborCol = col - (window / 2) - windowCol, 0;
                    neighborRow = row - (window / 2) - windowRow, 0;

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
                    gR = gaussian(inputFloat[neighborCol + neighborRow * cols] - inputFloat[col + row * cols], 0.0, sigmaR);
                    // Distance factor
                    gD = gaussian(distance(col, row, neighborCol, neighborRow), 0.0, sigmaD);

                    filteredPixel += inputFloat[neighborCol + neighborRow * cols] * (gR * gD);

                    wP += (gR * gD);
                }
            }
            output[col + row * cols] = filteredPixel / wP;
        }
    }

    (*outputFloat) = output;
}