#include "cudaLib.h"


/*
*   Constructor for CudaLib
*/
CudaLib::CudaLib(){
	 int deviceCount;

    cudaGetDeviceCount(&deviceCount);

	for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;

        cudaGetDeviceProperties(&deviceProp, dev);

        
    	
        if(dev == 0){
    		this->maxDevice = deviceProp.maxThreadsDim[dev];
        }
    }

}

/*
*   Destructor for CudaLib
*/
CudaLib::~CudaLib(){}

void CudaLib::loadOption(Option* option){

}

void CudaLib::loadBS(BS* bs){

}

void CudaLib::loadMonteCarlo(MonteCarlo* mc){

}
