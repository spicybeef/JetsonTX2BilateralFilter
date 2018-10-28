#include <stdio.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "main.h"

int main( int argc, char** argv )
{
    String inputImagePath("input.png");
    Mat image;

    // If we've passed in an image, use that one instead
    if( argc > 1)
    {
        inputImagePath = argv[1];
    }

    // Parse in the image
    image = cv::imread(inputImagePath, IMREAD_GRAYSCALE);

    // Check if that worked
    if (image.empty())
    {
        std::cout << "Error while attempting to open image" << std::endl;
        std::cin.get();
        return -1;
    }

    // Create window for display
    cv::namedWindow( "Display window", WINDOW_AUTOSIZE );
    // Show the image inside of it
    cv::imshow( "Display window", image );
    // Wait for a keystroke in the window
    cv::waitKey(0);

    return 0;
}