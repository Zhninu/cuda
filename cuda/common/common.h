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
	//memory
	template<typename T>
	static bool	mallocArray1D(T** ptr, vdim3 dim)
	{
		bool bMalloc = true;
		if (!ptr || *ptr)
			return false;

		unsigned long nSize = Common::calcDim(dim);
		unsigned long nBytes = nSize * sizeof(T);
		(*ptr) = (T*)malloc(nBytes);
		memset((*ptr), 0, nBytes);

		return bMalloc;
	}

	template<typename T>
	static bool freeArray1D(T** ptr) 
	{
		bool bFree = true;
		if (!ptr || !(*ptr))
			return false;

		if (*ptr)
		{
			free(*ptr);
			*ptr = NULL;
		}
		return bFree;
	}

	template<typename T>
	static bool mallocArray2D(T*** ptr, vdim3 dim) 
	{
		bool bMalloc = true;
		if (!ptr)
			return false;

		unsigned int height = dim.hei;
		unsigned int width = dim.col * dim.row;

		*ptr = new T*[dim.hei];
		for (unsigned int i = 0; i < height; i++)
		{
			(*ptr)[i] = new T[width];
		}
		return bMalloc;
	}

	template<typename T>
	static bool freeArray2D(T*** ptr, vdim3 dim)
	{
		bool bFree = true;
		if (!ptr)
			return false;

		unsigned int height = dim.hei;
		unsigned int width = dim.col * dim.row;

		for (unsigned int i = 0; i < height; i++)
		{
			delete (*ptr)[i];
		}
		delete[](*ptr);
		return bFree;
	}

	//construct
	template<typename T>
	static void constructArray(T* ptr, const unsigned long size)
	{
		if (!ptr)
			return;

		time_t tm;
		srand((unsigned)time(&tm));

		for (unsigned long i = 0; i < size; i++)
		{
			ptr[i] = (T)(rand() & 0xFF);
		}
	}

	template<typename T>
	static void constructArray2D(T*** ptr, unsigned long col, unsigned long row)
	{
		if (!ptr)
			return;

		time_t tm;
		srand((unsigned)time(&tm));

		for (unsigned int i = 0; i < row; i++)
		{
			for (unsigned int j = 0; j < col; j++)
			{
				(*ptr)[i][j] = (short)(rand() & 0xFF);
			}
		}
	}

	//common
	static Module findMode(char* str);
	//convert
	static void convertArray(SDS1D* a1d, SDS2D* a2d, cvArray type);
	//check
	static void campareResult(short* host, short* gpu, const unsigned long size);
	static void campareResult(short** hostArray, short** gpuArray, const vdim3 dim);
	static bool campareResult(int *ptr, const int n, const int x);
	//calculate
	static unsigned long calcDim(vdim3 dim);
	static unsigned long calcDim(vdim2 dim);
};

#endif
