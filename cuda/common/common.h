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

void initRandData(short* ip, const int size);
void campareResult(short* host, short* gpu, const int size);
int	 calcDimSize(vdim3 dim);

#endif
