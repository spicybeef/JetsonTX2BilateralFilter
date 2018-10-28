#include <stdio.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "main.h"

int main( int argc, char** argv )
{
    cv::String inputImagePath("input.png");
    cv::Mat inputImage;
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

    // Try running the CV bilateral filter on it
    cv::bilateralFilter(inputImage,outputImage,20,75,75);

    // Create window for display
    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    // Show the image inside of it
    cv::imshow("Output", outputImage);
    // Wait for a keystroke in the window
    cv::waitKey(0);

    return 0;
}
