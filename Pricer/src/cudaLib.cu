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

void CudaLib::loadOption(Option* opt){
        cudaError_t err;
      
        err = cudaMalloc((void **)&(opt->T_),sizeof(float));
        err = cudaMalloc((void **)&(opt->timeSteps_),sizeof(int));
        err = cudaMalloc((void **)&(opt->size_),sizeof(int));
        
        err = cudaMalloc((void **)&( (dynamic_cast<OptionBasket*>(opt))->strike_ ),sizeof(float));
        
        int sizePayoffCoeff = (dynamic_cast<OptionBasket*>(opt))->payoffCoeff_->size;
        double *arrayPayoffCoeff = (dynamic_cast<OptionBasket*>(opt))->payoffCoeff_->array;
        err = cudaMalloc((void **)&(sizePayoffCoeff),sizeof(int));
        err = cudaMalloc((void **)arrayPayoffCoeff ,sizeof(float)*sizePayoffCoeff);
     
}

void CudaLib::loadBS(BS* bs){

}

void CudaLib::loadMonteCarlo(MonteCarlo* mc){
        loadOption(mc->opt_);
        loadBS(mc->mod_);
        
        cudaError_t err;
        err = cudaMalloc((void **)&(this->h_),sizeof(float));
        err = cudaMalloc((void **)&(this->H_),sizeof(int));
        err = cudaMalloc((void **)&(this->samples_),sizeof(int));
}
