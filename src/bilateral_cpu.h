#pragma once

float distance(int x0, int y0, int x1, int y1);
float gaussian(float x, float mu, float sigma);
void bilateralNaiveCpu(float* inputFloat, float* outputFloat, int rows, int cols, uint32_t window, float sigmaD, float sigmaR);