#include <stdio.h>
#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "main.h"

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
 * @brief      Converts an OpenCV Mat object to a float array
 *
 * @param[in]  inputMat     The input Mat object
 * @param      outputFloat  The pointer to the output float array
 * @param[in]  rows         The number of rows in the Mat object
 * @param[in]  cols         The number of columns in the Mat object
 */
void matToFloatPtr(const cv::Mat* inputMat, float** outputFloat, int rows, int cols)
{
    // Instantiate new output float array
    float* output = static_cast<float*>(malloc(rows * cols * sizeof(float)));

    // Iterate through input mat and copy values into float array
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            output[col + row * cols] = inputMat->at<float>(row, col);
        }
    }

    (*outputFloat) = output;
}

/**
 * @brief      Converts a float array to a Mat object
 *
 * @param[in]  inputFloat  The input float array
 * @param      outputMat   The pointer to the output Mat object
 * @param[in]  rows        The number of rows in the Mat object 
 * @param[in]  cols        The number of columns in the Mat object
 */
void floatPtrToMat(const float* inputFloat, cv::Mat** outputMat, int rows, int cols)
{
    cv::Mat* output = new cv::Mat(rows, cols, CV_32F, cv::Scalar(0.5));

    // Iterate through input float and copy values into float Mat
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            output->at<float>(row, col) = inputFloat[col + row * cols];
        }
    }

    (*outputMat) = output;
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
    float* output = static_cast<float*>(malloc(rows * cols * sizeof(float)));

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

                    if (neighborCol < 0)
                    {
                        neighborCol = 0;
                    }
                    if (neighborRow < 0)
                    {
                        neighborRow = 0;
                    }

                    // Intensity
                    gR = gaussian(inputFloat[neighborCol + neighborRow * cols] - inputFloat[col + row * cols], 0.0, sigmaR);
                    // Distance
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

using namespace std::chrono;

int main( int argc, char** argv )
{
    cv::String inputImagePath("input.png");

    // If we've passed in an image, use that one instead
    if( argc > 1)
    {
        inputImagePath = argv[1];
    }
    
    // Parse in the image
    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);

    // Check if that worked
    if (inputImage.empty())
    {
        std::cout << "Error while attempting to open image" << std::endl;
        std::cin.get();
        return -1;
    }

    // Convert input to float in the range of 0.0 to 1.0
    cv::Mat inputImageFloat;
    inputImage.convertTo(inputImageFloat, CV_32F, 1.0/255.0, 0.0);
    
    // Try running the CV bilateral filter on it
    cv::Mat outputImageCv;
    high_resolution_clock::time_point t1Cv = high_resolution_clock::now();
    cv::bilateralFilter(inputImageFloat,outputImageCv, 10, 20, 30);
    high_resolution_clock::time_point t2Cv = high_resolution_clock::now();
    auto durationCv = duration_cast<milliseconds>(t2Cv - t1Cv).count();

    // Go from mat to pointer, do bilateral, then to go to mat again
    float * floatIntermediate;
    float * floatProcessed;
    cv::Mat * outputImagePtr;
    matToFloatPtr(&inputImageFloat, &floatIntermediate, inputImage.rows, inputImage.cols);
    high_resolution_clock::time_point t1Naive = high_resolution_clock::now();
    bilateralNaive(floatIntermediate, &floatProcessed, inputImage.rows, inputImage.cols, 10, 20, 30);
    high_resolution_clock::time_point t2Naive = high_resolution_clock::now();
    auto durationNaive = duration_cast<milliseconds>(t2Naive - t1Naive).count();
    floatPtrToMat(floatProcessed, &outputImagePtr, inputImage.rows, inputImage.cols);

    std::cout << "OpenCV bilateral took: " << durationCv << " ms" << std::endl;
    std::cout << "Naive bilateral took: " << durationNaive << " ms" << std::endl;

    // Original Version
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", inputImage);
    // CV Version
    cv::namedWindow("Output OpenCV", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output OpenCV", outputImageCv);
    // Naive Version
    cv::namedWindow("Output Naive", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output Naive", *outputImagePtr);

    // Wait for a keystroke in the window
    cv::waitKey(0);

    return 0;
}
