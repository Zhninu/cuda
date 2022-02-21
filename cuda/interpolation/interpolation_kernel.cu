#include "interpolation_kernel.cuh"

__global__ void interpolationKernel(short * vol, short * interp, SDSF3* pts, int size, int col, int row, int hei) 
{
	unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	unsigned long threadId = blockId * blockDim.x + threadIdx.x;

	SDSF3 ptInterp = pts[threadId];
	int CTV_OFFSET = 1024;
	short nVal = -CTV_OFFSET;
	int nx = (int)ptInterp.q1;
	int ny = (int)ptInterp.q2;
	int nz = (int)ptInterp.q3;

	if (nx < 0 || nx > col - 1 || ny < 0 || ny > row - 1 || nz < 0 || nz > hei - 1)
	{
		interp[threadId] = nVal;
		return;
	}

	int nz1 = nz == hei - 1 ? nz : nz + 1;
	int ny1 = ny == row - 1 ? ny : ny + 1;
	int nx1 = nx == col - 1 ? nx : nx + 1;

	int nyy = ny * col;
	int nyy1 = ny1 * col;

	int nyz = nz * col * row;
	int nyz1 = nz1 * col * row;
	float fv1 = vol[nyy + nx + nyz];
	float fv2 = vol[nyy + nx1 + nyz];
	float fv3 = vol[nyy1 + nx1 + nyz];
	float fv4 = vol[nyy1 + nx + nyz];
	float fv5 = vol[nyy + nx + nyz1];
	float fv6 = vol[nyy + nx1 + nyz1];
	float fv7 = vol[nyy1 + nx1 + nyz1];
	float fv8 = vol[nyy1 + nx + nyz1];

	float fsx = ptInterp.q1 - (float)nx;
	float fsy = ptInterp.q2 - (float)ny;
	float fsz = ptInterp.q3 - (float)nz;
	float fz1 = fv1 + (fv5 - fv1)*fsz;
	float fz4 = fv4 + (fv8 - fv4)*fsz;
	float fx1 = fz1 + (fv2 + (fv6 - fv2)*fsz - fz1)*fsx;
	nVal = fx1 + (fz4 + (fv3 + (fv7 - fv3)*fsz - fz4)*fsx - fx1)*fsy + 0.5f;
	nVal = nVal > -CTV_OFFSET ? nVal : -CTV_OFFSET;
	nVal = nVal < 3072 ? nVal : 3071;

	interp[threadId] = nVal;
}