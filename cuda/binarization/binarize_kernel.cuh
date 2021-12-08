#include "stdafx.h"
#include "typedef.h"

__global__ void binarizeKernel(short * volumedata, short * binarydata, int size, int thresh, int maxval);