#pragma once

#ifndef _COMMON_H_
#define _COMMON_H_

#include "helper_cuda.h"
#include "log_message.h"

#define CheckCudaErrors(val)    cudaCheck ( (val), #val, __FILE__, __LINE__)

template<typename T>
void cudaCheck(T result, char const * const func, const char * const file, int const line, bool bAbort = true, const char *pModuleName = "GPUError")
{
	if (result != cudaSuccess)
	{
		log_error(pModuleName, LogFormatA_A("CUDA error at %s:%d code=%d(%s) \"%s\".",
			file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func).c_str());
		log_error(pModuleName, LogFormatA_A("Error name is \"%s\". Error string is \"%s\".",
			cudaGetErrorName(result), cudaGetErrorString(result)).c_str());

		cudaDeviceReset();
		if (bAbort)
		{
			exit(result);
		}
	}
}

class Common
{
public:
	static Module findModeType(char* str);
	static bool	mallocVolume(SDS3D** vol, vdim3 dim);
	static bool freeVolume(SDS3D** vol);
	static void allocArray2D(short** array2D, vdim3 dim);
	static void freeArray2D(short** array2D, vdim3 dim);
	static void initArray2D(short** array2D, vdim3 dim);
	static void convertArray2D(SDS1D* array1D, SDS2D* array2D, cvArray type);
	static void initRandData(short* ip, const unsigned long size);
	static void campareResult(short* host, short* gpu, const unsigned long size);
	static void campareResult(short** hostArray, short** gpuArray, const vdim3 dim);
	static bool campareResult(int *data, const int n, const int x);
	static unsigned long calcDim(vdim3 dim);
	static unsigned long calcDim(vdim2 dim);
};

#endif
