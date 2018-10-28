#include <stdio.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "main.h"

int main( int argc, char** argv )
{
    cv::String inputImagePath("input.png");
    cv::Mat image;

    // If we've passed in an image, use that one instead
    if( argc > 1)
    {
        inputImagePath = argv[1];
    }

    // Parse in the image
    image = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);

    // Check if that worked
    if (image.empty())
    {
        std::cout << "Error while attempting to open image" << std::endl;
        std::cin.get();
        return -1;
    }

    // Create window for display
    cv::namedWindow("Output", cv::WINDOW_NORMAL);
    cv::resizeWindows("Output", 1024, 768);
    // Show the image inside of it
    cv::imshow("Output", image);
    // Wait for a keystroke in the window
    cv::waitKey(0);

    return 0;
}