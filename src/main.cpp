#include <stdio.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "main.h"

void matToFloatPtr(cv::Mat ** inputMat, const float ** outputFloat, int rows, int cols)
{

}

void floatPtrToMat(const float ** inputFloat, cv::Mat ** outputMat, int rows, int cols)
{
    cv::Mat * output = new cv::Mat(rows, cols, CV_32F, cv::Scalar(0.5));

    (*outputMat) = output;
}

void bilateralNaive(const float * input, const float * output, int rows, int cols, uint32_t window, float sigmaD, float sigmalR)
{

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
    cv::bilateralFilter(inputImageFloat,outputImageCv,20,50,50);
    // From pointer
    cv::Mat * outputImagePtr;
    floatPtrToMat(NULL, &outputImagePtr, inputImage.rows, inputImage.cols);

    // Original Version
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", inputImage);
    // CV Version
    cv::namedWindow("Output OpenCV", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output OpenCV", outputImageCv);
    // Naive Version
    cv::namedWindow("Output OpenCV", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output Naive", *outputImagePtr);

    // Wait for a keystroke in the window
    cv::waitKey(0);

    return 0;
}