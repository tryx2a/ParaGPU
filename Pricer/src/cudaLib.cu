#include "cudaLib.h"


/*
*   Constructor for CudaLib
*/
CudaLib::CudaLib(MonteCarlo* mc){
  int deviceCount;

  cudaGetDeviceCount(&deviceCount);

  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;  
    cudaGetDeviceProperties(&deviceProp, dev);
    if(dev == 0){
      this->maxDevice = deviceProp.maxThreadsDim[dev];
    }
  }
  
  this->allocMonteCarlo(mc);
}

/*
*   Destructor for CudaLib
*/
CudaLib::~CudaLib(){
  
}

void CudaLib::allocOption(Option* opt){
        cudaError_t err;   
        
        //Allocation de la maturité
        err = cudaMalloc((void **) &(this->T),sizeof(float));
        if(err != cudaSuccess){
          printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
          exit(EXIT_FAILURE);
        }
        
        //Allocation du nombre de pas de constatation
        err = cudaMalloc((void **) &(this->TimeSteps),sizeof(int));
        if(err != cudaSuccess){
          printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
          exit(EXIT_FAILURE);
        }
        
        //Allocation du strike pour une option basket
        err = cudaMalloc((void **) &(this->strike),sizeof(float));
        if(err != cudaSuccess){
          printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
          exit(EXIT_FAILURE);
        }
        
        //Allocation de la taille du vecteur PayoffCoeff
        err = cudaMalloc((void **)&(this->size_payoffCoeff),sizeof(int));
        if(err != cudaSuccess){
          printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
          exit(EXIT_FAILURE);
        }
        
        //Allocation du tableau du vecteur PayoffCoeff
        err = cudaMalloc((void **) &(this->payoffCoeff) ,sizeof(double)*opt->size_);
        if(err != cudaSuccess){
          printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
          exit(EXIT_FAILURE);
        }
     
}

void CudaLib::allocBS(BS* bs){

    //Allocation de la taille de BS
    cudaError_t err = cudaMalloc( (void**) &(this->size), sizeof(int));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //Allocation du taux
    err = cudaMalloc( (void**) &(this->r), sizeof(float));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    // trend size
    err = cudaMalloc( (void**) &(this->size_trend), sizeof(int));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*trend array
    err = cudaMalloc( (void**) &(this->trend), sizeof(float)*bs->size_);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*rho_
    err = cudaMalloc( (void**) &(this->rho), sizeof(float));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*sigma_ size
    err = cudaMalloc( (void**) &(this->size_sigma), sizeof(int));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*sigma_ array
    err = cudaMalloc( (void**) &(this->sigma), sizeof(float)*bs->size_);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*spot_ size
    err = cudaMalloc( (void**) &(this->size_spot), sizeof(int));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*spot_ array
    err = cudaMalloc( (void**) &(this->spot), sizeof(float)*bs->size_);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*chol_ lig
    err = cudaMalloc( (void**) &(this->m), sizeof(int));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }
    
    //*chol_ col
    err = cudaMalloc( (void**) &(this->n), sizeof(int));
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

    //*chol_ array
    err = cudaMalloc( (void**) &(this->chol), sizeof(float)*bs->chol->m*bs->chol->n);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }
}

void CudaLib::allocMonteCarlo(MonteCarlo* mc){
        
        cudaError_t err;
        
        //Allocation du pas de différence fini     
        err = cudaMalloc((void **) &(this->h),sizeof(float));
        if(err != cudaSuccess){
          printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
          exit(EXIT_FAILURE);
        }
        
        //Allocation du nombre de date à couvrir
        err = cudaMalloc((void **) &(this->H),sizeof(int));
        if(err != cudaSuccess){
          printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
          exit(EXIT_FAILURE);
        }
        
        //Allocation du nombre de tour de la boucle MonteCarlo
        err = cudaMalloc((void **) &(this->samples),sizeof(int));
        if(err != cudaSuccess){
          printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
          exit(EXIT_FAILURE);
        }
        
        //Allocation de l'option
        allocOption(mc->opt_);
        //Allocation du modèle de Black&Scholes
        allocBS(mc->mod_);
             
}





