#include "stdafx.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "cuda_verify.cuh"
#include "cuda_keneral.cuh"
#include "../common/common.h"

CCudaVerify::CCudaVerify()
	:m_pMoudle(LOG_CUDA_VERIFY_MODULE)
{

}

CCudaVerify::~CCudaVerify()
{

}

int CCudaVerify::memcpyAsync(int argc, char **argv)
{
	if (argc < 2)
		return 0;

    int devID = 0;
    cudaDeviceProp deviceProps;

    log_info(m_pMoudle, LogFormatA_A("Memcpy async- starting").c_str());
    log_info(m_pMoudle, LogFormatA_A("[%s] - starting", argv[0]).c_str());

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);

    // get device name
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    log_info(m_pMoudle, LogFormatA_A("CUDA device [%s]", deviceProps.name).c_str());

    int n = 16 * 1024 * 1024;
    int nbytes = n * sizeof(int);
    int value = 0;
	bool nPinnedMem = argv[2] != NULL ? stricmp(argv[2], "sync") : true;

    // allocate host memory
    int *a = 0;
	if (!nPinnedMem)
		a = (int*)malloc(nbytes);
	else
		checkCudaErrors(cudaMallocHost((void**)&a, nbytes));
    memset(a,0,nbytes);

    // allocate device memory
    int *d_a = 0;
    checkCudaErrors(cudaMalloc((void**)&d_a, nbytes));
    checkCudaErrors(cudaMemset(d_a, 255, nbytes));

    // set kernel launch configuration
    dim3 threads = dim3(512, 1);
    dim3 blocks = dim3(n / threads.x, 1);

    // create cuda event handles
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
    m_stTimer.startTimer("cuda memcpy HtD async");
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
    m_stTimer.stopTimer("cuda memcpy HtD async");
    increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    m_stTimer.startTimer("cuda memcpy DtH async");
	cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    m_stTimer.stopTimer("cuda memcpy DtH async");
    cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);

    // have CPU do some work while waiting for stage 1 to finish
    unsigned long int counter=0;
    while (cudaEventQuery(stop) == cudaErrorNotReady)
    {
        counter++;
    }
    
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
    log_info(m_pMoudle, LogFormatA_A("time spent executing by the GPU: %.2f", gpu_time).c_str());
    log_info(m_pMoudle, LogFormatA_A("time spent by CPU in CUDA calls: %.2f", sdkGetTimerValue(&timer)).c_str());
    log_info(m_pMoudle, LogFormatA_A("CPU executed %lu iterations while waiting for GPU to finish", counter).c_str());

    // check the output for correctness
    bool bFinalResults = Common::campareResult(a, n, value);

    // release resources
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
	if (nPinnedMem == 0)
		free(a);
	else
		checkCudaErrors(cudaFreeHost(a));
    checkCudaErrors(cudaFree(d_a));

    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}