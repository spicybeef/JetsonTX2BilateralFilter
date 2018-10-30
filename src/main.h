#pragma once

void matToFloatPtr(cv::Mat ** inputMat, const float ** outputFloat, int rows, int cols);
void floatPtrToMat(const float ** inputFloat, cv::Mat ** outputMat, int rows, int cols);
void bilateralNaive(const float * input, const float * output, int rows, int cols, uint32_t window, float sigmaD, float sigmalR);