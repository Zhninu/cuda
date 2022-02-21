#include "stdafx.h"
#include "typedef.h"

__global__ void interpolationKernel(short * vol, short * interp,  SDSF3* pts, int size, int col, int row, int hei);