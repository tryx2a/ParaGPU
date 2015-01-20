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
    //Charge la taille de BS
    cudaError_t err = cudaMalloc( (void**) &(bs->size_), sizeof(int));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //Charge le taux
    err = cudaMalloc( (void**) &(bs->r_), sizeof(float));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    // trend size
    int sizeTrend = bs->trend->size;
    err = cudaMalloc( (void**) &(sizeTrend), sizeof(int));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*trend array
    double* arrayTrend = bs->trend->array;
    err = cudaMalloc( (void**) arrayTrend, sizeof(float)*sizeTrend);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*rho_
    err = cudaMalloc( (void**) &(bs->rho_), sizeof(float));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*sigma_ size
    int sizeSigma = bs->sigma_->size;
    err = cudaMalloc( (void**) &(sizeSigma), sizeof(int));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*sigma_ array
    double* arraySigma = bs->sigma_->array;
    err = cudaMalloc( (void**) arraySigma, sizeof(float)*sizeSigma);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*spot_ size
    int sizeSpot = bs->spot_->size;
    err = cudaMalloc( (void**) &(sizeSpot), sizeof(int));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*spot_ array
    double* arraySpot = bs->spot_->array;
    err = cudaMalloc( (void**) arraySpot, sizeof(float)*sizeSpot);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*chol_ lig
    int ligChol = bs->chol->m;
    err = cudaMalloc( (void**) &(ligChol), sizeof(int));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }
    
    //*chol_ col
    int colChol = bs->chol->n;
    err = cudaMalloc( (void**) &(colChol), sizeof(int));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*chol_ array
    double* arrayChol = bs->chol->array;
    err = cudaMalloc( (void**) arrayChol, sizeof(double)*ligChol*colChol);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }
}

void CudaLib::loadMonteCarlo(MonteCarlo* mc){
        loadOption(mc->opt_);
        loadBS(mc->mod_);
        
        cudaError_t err;
        err = cudaMalloc((void **)&(this->h_),sizeof(float));
        err = cudaMalloc((void **)&(this->H_),sizeof(int));
        err = cudaMalloc((void **)&(this->samples_),sizeof(int));
}
