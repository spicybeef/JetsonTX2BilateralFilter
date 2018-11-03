#include <stdio.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "main.h"

float distance(int x0, int y0, int x1, int y1)
{
    return static_cast<float>(sqrt( (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) ));
}

float gaussian(float x, float mu, float sigma)
{
    return static_cast<float>(exp(-((x - mu) * (x - mu))/(2 * sigma * sigma)) / (2 * M_PI * sigma * sigma));
}

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
    cv::bilateralFilter(inputImageFloat,outputImageCv, 10, 20, 30);
    // Go from mat to pointer, do bilateral, then to go to mat again
    float * floatIntermediate;
    float * floatProcessed;
    cv::Mat * outputImagePtr;
    matToFloatPtr(&inputImageFloat, &floatIntermediate, inputImage.rows, inputImage.cols);
    bilateralNaive(floatIntermediate, &floatProcessed, inputImage.rows, inputImage.cols, 10, 20, 30);
    floatPtrToMat(floatProcessed, &outputImagePtr, inputImage.rows, inputImage.cols);

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
