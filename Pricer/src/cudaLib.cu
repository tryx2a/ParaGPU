#include "cudaLib.h"
#include "optionBasket.h"


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
  
  /* Chargement des objets de type pnl dans la mémoire GPU */
  this->allocMonteCarlo(mc);
  this->memcpyMonteCarlo(mc);

  /// Initialise la grille et les dimensions de chaque bloc
  dim3 DimGrid(4,4,1);
  dim3 DimBlock(16,16,1);
}

/*
 *   Destructor for CudaLib
 */
CudaLib::~CudaLib(){
  cudaFree(this->trend);
  cudaFree(this->sigma);
  cudaFree(this->spot);
  cudaFree(this->chol);

  if(this->payoffCoeff != NULL){
    cudaFree(this->payoffCoeff);
  }
}

void CudaLib::allocOption(Option* opt){
  cudaError_t err;   

    //Allocation du tableau du vecteur PayoffCoeff
  err = cudaMalloc((void **) &(this->payoffCoeff) ,sizeof(double)*opt->size_);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

}

void CudaLib::allocBS(BS* bs){
  cudaError_t err;

    //*trend array
  err = cudaMalloc( (void**) &(this->trend), sizeof(float)*bs->size_);
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

    //*spot_ array
  err = cudaMalloc( (void**) &(this->spot), sizeof(float)*bs->size_);
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

    //Allocation de l'option
  allocOption(mc->opt_);
    //Allocation du modèle de Black&Scholes
  allocBS(mc->mod_);

}

void CudaLib::memcpyOption(Option* opt){
  cudaError_t err;   

    //Chargement en mémoire du tableau du vecteur PayoffCoeff
  OptionBasket *basket = dynamic_cast<OptionBasket *>(opt);
  err = cudaMemcpy(this->payoffCoeff, basket->payoffCoeff_->array, basket->payoffCoeff_->size*sizeof(float), cudaMemcpyHostToDevice);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }
}

void CudaLib::memcpyBS(BS* bs){
  cudaError_t err; 

    //*trend array
  err = cudaMemcpy(this->trend, bs->trend->array, bs->trend->size*sizeof(float), cudaMemcpyHostToDevice);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

    //*sigma_ array
  err = cudaMemcpy(this->sigma, bs->sigma_->array, bs->sigma_->size*sizeof(float), cudaMemcpyHostToDevice);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

    //*spot_ array
  err = cudaMemcpy(this->spot, bs->spot_->array, bs->spot_->size*sizeof(float), cudaMemcpyHostToDevice);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

    //*chol_ array
  err = cudaMemcpy(this->chol, bs->chol->array, bs->chol->m*bs->chol->n*sizeof(float), cudaMemcpyHostToDevice);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }
}

void CudaLib::memcpyMonteCarlo(MonteCarlo* mc){
    //Chargement en mémoire de l'option
  memcpyOption(mc->opt_);
    //Chargemen en mémoire du modèle de Black&Scholes
  memcpyBS(mc->mod_);  
}



