#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <arm_neon.h>
#include <memory.h>

// Making this the size of the window for now
#define SIMD_SIZE 10

// Duplication of exp implementation for assembly analysis. Wasn't able to look
// at the disassembly of built-in expf function because it made a call into a
// library I couldn't disassemble easily. This is taken directly out of the GNU
// C Library (glibc). For some reason this is faster than using the builtin expf
// function (???).
extern "C"
{
    static inline double_t roundtoint (double_t x)
    {
        return vget_lane_f64 (vrndn_f64 (vld1_f64 (&x)), 0);
    }
    static inline uint64_t converttoint (double_t x)
    {
        return vcvtnd_s64_f64 (x);
    }
    static inline double asdouble (uint64_t i)
    {
        union
        {
            uint64_t i;
            double f;
        }
        u = {i};
        return u.f;
    }
    #define EXP2F_TABLE_BITS 5
    #define EXP2F_POLY_ORDER 3
    extern const struct exp2f_data_michel
    {
      uint64_t tab[1 << EXP2F_TABLE_BITS];
      double shift_scaled;
      double poly[EXP2F_POLY_ORDER];
      double shift;
      double invln2_scaled;
      double poly_scaled[EXP2F_POLY_ORDER];
    } __exp2f_data_michel;
    #define N (1 << EXP2F_TABLE_BITS)
    const struct exp2f_data_michel __exp2f_data_michel =
    {
        /* tab[i] = uint(2^(i/N)) - (i << 52-BITS)
         used for computing 2^(k/N) for an int |k| < 150 N as
         double(tab[k%N] + (k << 52-BITS)) */
        .tab =
        {
            0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
            0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
            0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
            0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
            0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
            0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
            0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
            0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
        },
        .shift_scaled = 0x1.8p+52 / N,
        .poly = { 0x1.c6af84b912394p-5, 0x1.ebfce50fac4f3p-3, 0x1.62e42ff0c52d6p-1 },
        .shift = 0x1.8p+52,
        .invln2_scaled = 0x1.71547652b82fep+0 * N,
        .poly_scaled = { 0x1.c6af84b912394p-5/N/N/N, 0x1.ebfce50fac4f3p-3/N/N, 0x1.62e42ff0c52d6p-1/N },
    };
    #define InvLn2N __exp2f_data_michel.invln2_scaled
    #define T __exp2f_data_michel.tab
    #define C __exp2f_data_michel.poly_scaled

    float expf_michel (float x)
    {
        uint64_t ki, t;
        /* double_t for better performance on targets with FLT_EVAL_METHOD==2.  */
        double_t kd, xd, z, r, r2, y, s;

        xd = (double_t) x;

        /* x*N/Ln2 = k + r with r in [-1/2, 1/2] and int k.  */
        z = InvLn2N * xd;

        /* Round and convert z to int, the result is in [-150*N, 128*N] and
         ideally ties-to-even rule is used, otherwise the magnitude of r
         can be bigger which gives larger approximation error.  */
        kd = roundtoint(z);
        ki = converttoint(z);
        r = z - kd;

        /* exp(x) = 2^(k/N) * 2^(r/N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1) */
        t = T[ki % N];
        t += ki << (52 - EXP2F_TABLE_BITS);
        s = asdouble(t);
        z = C[0] * r + C[1];
        r2 = r * r;
        y = C[2] * r + 1;
        y = z * r2 + y;
        y = y * s;
        return (float) y;
    }
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
 * @brief      Calculates the Euclidean distance between an array of two points.
 *
 * @param[in]  x0          The x0 coordinate array
 * @param[in]  y0          The y0 coordinate array
 * @param[in]  x1          The x1 coordinate array
 * @param[in]  y1          The y1 coordinate array
 * @param      distResult  A pointer to the resulting distance array
 */
void distanceSimd(int* x0, int* y0, int* x1, int* y1, float* distResult)
{
    int tempX[SIMD_SIZE];
    int tempY[SIMD_SIZE];
    float tempXPow[SIMD_SIZE];
    float tempYPow[SIMD_SIZE];
    float tempSqrt[SIMD_SIZE];

    for(int i = 0; i < SIMD_SIZE; i++)
    {
        tempX[i] = x0[i] - x1[i];
    }

    for(int i = 0; i < SIMD_SIZE; i++)
    {
        tempY[i] = y0[i] - y1[i];
    }

    for(int i = 0; i < SIMD_SIZE; i++)
    {
        tempXPow[i] = tempX[i] * tempX[i];
    }

    for(int i = 0; i < SIMD_SIZE; i++)
    {
        tempYPow[i] = tempY[i] * tempY[i];
    }

    for(int i = 0; i < SIMD_SIZE; i++)
    {
        tempSqrt[i] = tempXPow[i] + tempYPow[i];        
    }

    for(int i = 0; i < SIMD_SIZE; i++)
    {
        distResult[i] = sqrtf(tempSqrt[i]);
    }

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
 * @brief      Calculates the one-dimensional Gaussian for an array of given
 *             points
 *
 * @param      x            The pointer to the points to calculate the Gaussian
 * @param[in]  sigma        The standard deviation of the distribution
 * @param      gaussResult  An array to the gaussian results
 */
void gaussianSimd(float* x, float sigma, float* gaussResult)
{
    float tempXPow[SIMD_SIZE];

    float alpha = 2.0 * sigma * sigma;
    float beta = alpha * M_PI;

    for(int i = 0; i < SIMD_SIZE; i++)
    {
        tempXPow[i] = x[i] * x[i];
    }

    for(int i = 0; i < SIMD_SIZE; i++)
    {
        tempXPow[i] = -tempXPow[i] / alpha;
    }

    for(int i = 0; i < SIMD_SIZE; i++)
    {
        gaussResult[i] = expf_michel(tempXPow[i]) / beta;
    }
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

/**
 * @brief      An optimized version of the CPU bilateral filter. Optmizations
 *             include using OpenMP for threading of the main loop and changing
 *             memory access to be sequential by turning the outer loops into a
 *             single loop. Also naively implements 10-wide SIMD instructions.
 *
 * @param      inputFloat   The input float array
 * @param      outputFloat  The output float array
 * @param[in]  rows         The number of rows in the image
 * @param[in]  cols         The number of columns in the image
 * @param[in]  window       The window to use in the filter
 * @param[in]  sigmaD       The distance parameter
 * @param[in]  sigmaR       The intensity parameter
 */
void bilateralOptimizedCpuSimd(float* inputFloat, float* outputFloat, int rows, int cols, uint32_t window, float sigmaD, float sigmaR)
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
        float distances[SIMD_SIZE];
        int currentCol[SIMD_SIZE];
        int currentRow[SIMD_SIZE];
        int neighbourCols[SIMD_SIZE];
        float neighbourPixels[SIMD_SIZE];
        float intensityDiffs[SIMD_SIZE];
        float gRs[SIMD_SIZE];
        float gDs[SIMD_SIZE];
        int row = pixel / cols;
        int col = pixel % cols;

        for(int i = 0; i < SIMD_SIZE; i++)
        {
            currentCol[i] = col;
            currentRow[i] = row;
        }

        for(int i = 0; i < window; i++)
        {
            neighbourCols[i] = col - (window / 2) - i;
            // Prevent us indexing into regions that don't exist
            if (neighbourCols[i] < 0)
            {
                neighbourCols[i] = 0;
            }
        }

        for(int i = 0; i < window; i++)
        {
            neighbourPixels[i] = inputFloat[neighbourCols[i] + row * cols];
        }

        for(int i = 0; i < window; i++)
        {
           intensityDiffs[i] = neighbourPixels[i] - inputFloat[pixel]; 
        }

        // Intensity factor
        gaussianSimd(intensityDiffs, sigmaR, gRs);
        // Distances
        distanceSimd(currentCol, currentRow, neighbourCols, currentRow, distances);
        // Distance factor
        gaussianSimd(distances, sigmaD, gDs);

        for(int i = 0; i < window; i++)
        {
            filteredPixel += neighbourPixels[i] * gRs[i] * gDs[i];
        }

        for(int i = 0; i < window; i++)
        {
            wP += gRs[i] * gDs[i];
        }

        buffer[pixel] = filteredPixel / wP;
    }

    #pragma omp parallel for
    for (int pixel = 0; pixel < totalPixels; pixel++)
    {
        float filteredPixel = 0;
        float wP = 0;
        float distances[SIMD_SIZE];
        int currentCol[SIMD_SIZE];
        int currentRow[SIMD_SIZE];
        int neighbourRows[SIMD_SIZE];
        float neighbourPixels[SIMD_SIZE];
        float intensityDiffs[SIMD_SIZE];
        float gRs[SIMD_SIZE];
        float gDs[SIMD_SIZE];
        int row = pixel / cols;
        int col = pixel % cols;

        for(int i = 0; i < SIMD_SIZE; i++)
        {
            currentCol[i] = col;
            currentRow[i] = row;
        }

        for(int i = 0; i < window; i++)
        {
            neighbourRows[i] = row - (window / 2) - i;
            // Prevent us indexing into regions that don't exist
            if (neighbourRows[i] < 0)
            {
                neighbourRows[i] = 0;
            }
        }

        for(int i = 0; i < window; i++)
        {
            neighbourPixels[i] = buffer[col + neighbourRows[i] * cols];
        }

        for(int i = 0; i < window; i++)
        {
           intensityDiffs[i] = neighbourPixels[i] - buffer[pixel]; 
        }

        // Intensity factor
        gaussianSimd(intensityDiffs, sigmaR, gRs);
        // Distances
        distanceSimd(currentCol, currentRow, currentCol, neighbourRows, distances);
        // Distance factor
        gaussianSimd(distances, sigmaD, gDs);

        for(int i = 0; i < window; i++)
        {
            filteredPixel += neighbourPixels[i] * gRs[i] * gDs[i];
        }

        for(int i = 0; i < window; i++)
        {
            wP += gRs[i] * gDs[i];
        }

        outputFloat[pixel] = filteredPixel / wP;
    }
}