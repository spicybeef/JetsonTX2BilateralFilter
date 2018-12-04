#pragma once

void bilateralNaiveCpu(float* inputFloat, float* outputFloat, int rows, int cols, uint32_t window, float sigmaD, float sigmaR);
void bilateralOptimizedCpu(float* inputFloat, float* outputFloat, int rows, int cols, uint32_t window, float sigmaD, float sigmaR);