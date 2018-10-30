#include <stdio.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "main.h"

// void bilateralNaive(std input, cv::Mat & output, uint32_t window, float sigmaD, float sigmalR)
// {

// }

int main( int argc, char** argv )
{
    cv::String inputImagePath("input.png");
    cv::Mat inputImage;
    cv::Mat inputImageFloat;
    cv::Mat outputImage;

    // If we've passed in an image, use that one instead
    if( argc > 1)
    {
        inputImagePath = argv[1];
    }
    
    // Parse in the image
    inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);

    // Check if that worked
    if (inputImage.empty())
    {
        std::cout << "Error while attempting to open image" << std::endl;
        std::cin.get();
        return -1;
    }

    // Convert input to float
    inputImage.convertTo(inputImageFloat, CV_32F, 1.0/255.0);

    // Try running the CV bilateral filter on it
    cv::bilateralFilter(inputImageFloat,outputImage,20,50,50);

    // Create window for display
    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    // Show the image inside of it
    cv::imshow("Output", outputImage, c);
    // Wait for a keystroke in the window
    cv::waitKey(0);

    return 0;
}