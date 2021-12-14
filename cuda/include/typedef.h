#pragma once
#ifndef _TYPRDEF_H_
#define _TYPRDEF_H_

#define EC_OK	0
#define EC_ERR	1

/* module log info*/
#define LOG_BINARIZE_ENGINE_MODULE			"BinarizeEngine"
#define LOG_INTERPOLATION_ENGINE_MODULE		"InterpolationEngine"
#define LOG_CUDA_VERIFY_MODULE				"CudaVerify"

typedef enum tagModule 
{
	Module_All = 0,
	Module_Binary,
	Module_CudaVerify
}Module;

typedef enum tagTimerType
{
	timer_none = 0,
	timer_time,				//C系统调用, <1s
	timer_clock,			//C系统调用, <10ms
	timer_ApiTime,			//Windows API, <1ms, timeGetTime
	timer_QPCounter,		//Windows API, <<0.1ms, QueryPerformanceCounter
	timer_TickCount,		//CWindows API, <1ms, GetTickCount
}timer;

typedef enum tagLogLevel
{
	level_debug = 0,
	level_info,
	level_warning,
	level_error,
}LogLevel;

typedef enum tagConvertArrayType 
{
	cvArray_None = 0,
	cvArray_1DTo2D,
	cvArray_2DTo1D,
}cvArray;

struct vdim2 
{
	unsigned int col;
	unsigned int row;
	vdim2() {}
	vdim2(unsigned int vcol, unsigned int vrow) : col(vcol), row(vrow){}
};

struct vdim3
{
	unsigned int col; 
	unsigned int row;
	unsigned int hei;
	vdim3() {}
	vdim3(unsigned int vcol, unsigned int vrow, unsigned int vhei) : col(vcol), row(vrow), hei(vhei) {}
};

typedef struct tagSDSF3
{
	float q1;
	float q2;
	float q3;
}SDSF3;

typedef struct tagSDS1D
{
	short*	data;
	unsigned long size;
}SDS1D;

typedef struct tagSDS2D
{
	short**	data;
	vdim2	dim;
}SDS2D;

typedef struct tagSDS3D
{
	short*	data;
	vdim3	dim;
}SDS3D;

typedef struct tagBinarySDS3D : public SDS3D
{
	int thresh;
	int maxval;
}binSDS3D;

typedef struct tagInterpSDS3D : public SDS3D
{
	SDSF3* interp;
}interSDS3D;

#endif // _TYPRDEF_H_