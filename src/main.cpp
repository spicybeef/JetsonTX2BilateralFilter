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

// Replication of exp implementation for assembly analysis
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
    float* floatProcessedOptimizedCpu = new float [inputImage.rows * inputImage.cols];
    cv::Mat* outputImagePtrNaiveCpu;
    cv::Mat* outputImagePtrNaiveGpu;
    cv::Mat* outputImagePtrOptimizedCpu;

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

    // Optimized CPU
    matToFloatPtr(&inputImageFloat, &floatIntermediate, inputImage.rows, inputImage.cols);
    high_resolution_clock::time_point t1OptimizedCpu = high_resolution_clock::now();
    bilateralOptimizedCpu(floatIntermediate, floatProcessedOptimizedCpu, inputImage.rows, inputImage.cols, windowSize, sigmaD, sigmaR);
    high_resolution_clock::time_point t2OptimizedCpu = high_resolution_clock::now();
    auto durationOptimizedCpu = duration_cast<milliseconds>(t2OptimizedCpu - t1OptimizedCpu).count();
    floatPtrToMat(floatProcessedOptimizedCpu, &outputImagePtrOptimizedCpu, inputImage.rows, inputImage.cols);
    std::cout << "Optimized CPU bilateral took: " << durationOptimizedCpu << " ms" << std::endl;

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
    // Optimized CPU Version
    cv::namedWindow("Output Optimized CPU", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output Optimized CPU", *outputImagePtrOptimizedCpu);
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
