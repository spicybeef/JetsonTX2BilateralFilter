#include <stdio.h>
#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "bilateral_cpu.h"
#include "bilateral_gpu.h"

#define SHOW_OUTPUT

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

using namespace std::chrono;

int main( int argc, char** argv )
{
    cv::String inputImagePath("input.png");
    int windowSize = 10;
    int sigmaD = 20;
    int sigmaR = 5;

    // If we've passed in an image, use that one instead
    if (argc > 1)
    {
        inputImagePath = argv[1];
        windowSize = std::stoi(argv[2]);
        sigmaD = std::stoi(argv[3]);
        sigmaR = std::stoi(argv[4]);
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

    std::cout << "sizeof(float) = " << sizeof(float) << std::endl;
    std::cout << "Image size is " << inputImage.cols << " x " << inputImage.rows << std::endl;
    std::cout << "Total processed size for grayscale image is: " << inputImage.cols * inputImage.rows * sizeof(float) << " bytes" << std::endl;
    std::cout << "Using window size of: " << windowSize << std::endl;
    std::cout << "Using sigmaD size of: " << sigmaD << std::endl;
    std::cout << "Using sigmaR size of: " << sigmaR << std::endl;

    // Convert input to float in the range of 0.0 to 1.0
    cv::Mat inputImageFloat;
    inputImage.convertTo(inputImageFloat, CV_32F, 1.0/255.0, 0.0);
    
    // Original Version
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", inputImage);

    // Try running the CV bilateral filter on it
    cv::Mat outputImageCv;
    high_resolution_clock::time_point t1Cv = high_resolution_clock::now();
    cv::bilateralFilter(inputImageFloat,outputImageCv, windowSize, sigmaD, sigmaR);
    high_resolution_clock::time_point t2Cv = high_resolution_clock::now();
    auto durationCv = duration_cast<milliseconds>(t2Cv - t1Cv).count();
    std::cout << "OpenCV bilateral took: " << durationCv << " ms" << std::endl;

    // Go from mat to pointer, do bilateral, then to go to mat again
    float* floatIntermediate;
    float* floatProcessedNaiveCpu = new float [inputImage.rows * inputImage.cols];
    float* floatProcessedNaiveGpu = new float [inputImage.rows * inputImage.cols];
    cv::Mat* outputImagePtrNaiveCpu;
    cv::Mat* outputImagePtrNaiveGpu;

    // Naive CPU
    matToFloatPtr(&inputImageFloat, &floatIntermediate, inputImage.rows, inputImage.cols);
    high_resolution_clock::time_point t1NaiveCpu = high_resolution_clock::now();
    bilateralNaiveCpu(floatIntermediate, floatProcessedNaiveCpu, inputImage.rows, inputImage.cols, windowSize, sigmaD, sigmaR);
    high_resolution_clock::time_point t2NaiveCpu = high_resolution_clock::now();
    auto durationNaiveCpu = duration_cast<milliseconds>(t2NaiveCpu - t1NaiveCpu).count();
    floatPtrToMat(floatProcessedNaiveCpu, &outputImagePtrNaiveCpu, inputImage.rows, inputImage.cols);
    std::cout << "Naive CPU bilateral took: " << durationNaiveCpu << " ms" << std::endl;

    // Naive GPU
    matToFloatPtr(&inputImageFloat, &floatIntermediate, inputImage.rows, inputImage.cols);
    high_resolution_clock::time_point t1NaiveGpu = high_resolution_clock::now();
    bilateralNaiveGpu(floatIntermediate, floatProcessedNaiveGpu, inputImage.rows, inputImage.cols, windowSize, sigmaD, sigmaR);
    high_resolution_clock::time_point t2NaiveGpu = high_resolution_clock::now();
    auto durationNaiveGpu = duration_cast<milliseconds>(t2NaiveGpu - t1NaiveGpu).count();
    floatPtrToMat(floatProcessedNaiveGpu, &outputImagePtrNaiveGpu, inputImage.rows, inputImage.cols);
    std::cout << "Naive GPU bilateral took: " << durationNaiveGpu << " ms" << std::endl;

#if defined(SHOW_OUTPUT)
    // CV Version
    cv::namedWindow("Output OpenCV", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output OpenCV", outputImageCv);
    // Naive CPU Version
    cv::namedWindow("Output Naive CPU", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output Naive CPU", *outputImagePtrNaiveCpu);
    // Naive GPU Version
    cv::namedWindow("Output Naive GPU", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output Naive GPU", *outputImagePtrNaiveGpu);
#endif // defined(SHOW_OUTPUT)

    // Wait for escape
    while(1)
    {
        int k = cv::waitKey(33);

        if (k == 27)
        {
            break;
        }
    }

    return 0;
}
