#include "cudaLib.h"
#include "../Options/utils.h"


/*
 *   Constructor for CudaLib
 */
CudaLib::CudaLib(MonteCarlo* mc){
  
  /*
   * Calcul du nombre de thread max disponible
   */
  int deviceCount;

  cudaGetDeviceCount(&deviceCount);

  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;  
    cudaGetDeviceProperties(&deviceProp, dev);
    if(dev == 0){
      this->maxDevice = deviceProp.maxThreadsDim[dev];
    }
  }
  
  /*
   * Gestion du "Strike" des options
   */
  int idOpt = mc->opt_->id_;

  if(idOpt == 1){ //Option Asian
    this->strike = (dynamic_cast<OptionAsian *>(mc->opt_))->strike_;
  }
  else if(idOpt == 2){ //Option Barrière
    this->strike = (dynamic_cast<OptionBarrier *>(mc->opt_))->strike_;
  }
  else if(idOpt == 3){ //Option Barrière Basse
    this->strike = (dynamic_cast<OptionBarrierLow *>(mc->opt_))->strike_;
  }
  else if(idOpt == 4){ //Option Barrière Haute
    this->strike = (dynamic_cast<OptionBarrierUp *>(mc->opt_))->strike_;
  }
  else if(idOpt == 5){ //Option Basket
    this->strike = (dynamic_cast<OptionBasket *>(mc->opt_))->strike_;
  }
  else if(idOpt == 6){ //Option Performance
    //Cette option n'a pas de "Strike"
    this->strike = 0.0;
  }
  else{ //Ne devrait jamais arriver
    exit(EXIT_FAILURE);
  }

  /* On aligne le nombre de samples sur un multiple de maxDevice */
  int nbTourModifie = (int)(mc->samples_/this->maxDevice) * this->maxDevice;
  mc->samples_ = nbTourModifie;

  /* Allocation et chargement des objets de type pnl dans la mémoire GPU */
  this->allocMonteCarlo(mc);
  this->memcpyMonteCarlo(mc);

}

/*
 *   Destructor for CudaLib
 */
CudaLib::~CudaLib(){
  cudaFree(this->trend);
  cudaFree(this->sigma);
  cudaFree(this->spot);
  cudaFree(this->chol);

  cudaFree(this->tabPath);
  cudaFree(this->tabPrice);
  cudaFree(this->tabVar);

  if(this->payoffCoeff != NULL){
    cudaFree(this->payoffCoeff);
  }

  if(this->lowerBarrier != NULL){
    cudaFree(this->lowerBarrier);
  }

  if(this->upperBarrier != NULL){
    cudaFree(this->upperBarrier);
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

    //Allocation du tableau du vecteur pour la barrière basse
  err = cudaMalloc((void **) &(this->lowerBarrier) ,sizeof(double)*opt->size_);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

   //Allocation du tableau du vecteur pour la barrière haute
  err = cudaMalloc((void **) &(this->upperBarrier) ,sizeof(double)*opt->size_);
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

    //Allocation du tableau contenant une une matrice path pour chaque device
  cudaError_t err;

    //tabPath
  err = cudaMalloc( (void**) &(this->tabPath), sizeof(float)*mc->samples_*(mc->opt_->TimeSteps_ + 1)*mc->mod_->size_);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

   //tabPrice
  err = cudaMalloc( (void**) &(this->tabPrice), sizeof(float)*mc->samples_);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

   //tabVar
  err = cudaMalloc( (void**) &(this->tabVar), sizeof(float)*mc->samples_);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

}

void CudaLib::memcpyOption(Option* opt){
  cudaError_t err;   
  int idOpt = opt->id_;

  if(idOpt == 1){ //Option Asian
    //Rien à copier pour une telle option
  }

  else if(idOpt == 2){ //Option Barrière
    OptionBarrier *barrier = dynamic_cast<OptionBarrier *>(opt);

      //Chargement en mémoire du tableau du vecteur PayoffCoeff
    err = cudaMemcpy(this->payoffCoeff, utils::convertPnlVectToFloat(barrier->payoffCoeff_->array,barrier->payoffCoeff_->size), barrier->payoffCoeff_->size*sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

      //Chargement en mémoire du tableau du vecteur LowerBarrier
    err = cudaMemcpy(this->lowerBarrier, utils::convertPnlVectToFloat(barrier->lowerBarrier_->array,barrier->lowerBarrier_->size), barrier->lowerBarrier_->size*sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

      //Chargement en mémoire du tableau du vecteur UpperBarrier
    err = cudaMemcpy(this->upperBarrier, utils::convertPnlVectToFloat(barrier->upperBarrier_->array,barrier->upperBarrier_->size), barrier->upperBarrier_->size*sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

  }

  else if(idOpt == 3){ //Option Barrière basse
    OptionBarrierLow *barrierLow = dynamic_cast<OptionBarrierLow *>(opt);

      //Chargement en mémoire du tableau du vecteur PayoffCoeff
    err = cudaMemcpy(this->payoffCoeff, utils::convertPnlVectToFloat(barrierLow->payoffCoeff_->array,barrierLow->payoffCoeff_->size), barrierLow->payoffCoeff_->size*sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

      //Chargement en mémoire du tableau du vecteur LowerBarrier
    err = cudaMemcpy(this->lowerBarrier, utils::convertPnlVectToFloat(barrierLow->lowerBarrier_->array,barrierLow->lowerBarrier_->size), barrierLow->lowerBarrier_->size*sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }
  }

  else if(idOpt == 4){ //Option Barrière haute
    OptionBarrierUp *barrierUp = dynamic_cast<OptionBarrierUp *>(opt);

      //Chargement en mémoire du tableau du vecteur PayoffCoeff
    err = cudaMemcpy(this->payoffCoeff, utils::convertPnlVectToFloat(barrierUp->payoffCoeff_->array,barrierUp->payoffCoeff_->size), barrierUp->payoffCoeff_->size*sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }

      //Chargement en mémoire du tableau du vecteur UpperBarrier
    err = cudaMemcpy(this->upperBarrier, utils::convertPnlVectToFloat(barrierUp->upperBarrier_->array,barrierUp->upperBarrier_->size), barrierUp->upperBarrier_->size*sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }
  }

  else if(idOpt == 5){ //Option Basket
    OptionBasket *basket = dynamic_cast<OptionBasket *>(opt);

      //Chargement en mémoire du tableau du vecteur PayoffCoeff
    err = cudaMemcpy(this->payoffCoeff, utils::convertPnlVectToFloat(basket->payoffCoeff_->array,basket->payoffCoeff_->size), basket->payoffCoeff_->size*sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }
  }

  else if(idOpt == 6){ //Option Performance
    OptionPerformance *performance = dynamic_cast<OptionPerformance *>(opt);

      //Chargement en mémoire du tableau du vecteur PayoffCoeff
    err = cudaMemcpy(this->payoffCoeff, utils::convertPnlVectToFloat(performance->payoffCoeff_->array,performance->payoffCoeff_->size), performance->payoffCoeff_->size*sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
    }
  }

  else{ //Cette id n'existe pas, ne doit jamais arriver
    exit(EXIT_FAILURE);
  }
  
}

void CudaLib::memcpyBS(BS* bs){
  cudaError_t err; 

    //*trend array
  err = cudaMemcpy(this->trend, utils::convertPnlVectToFloat(bs->trend->array,bs->size_), bs->trend->size*sizeof(float), cudaMemcpyHostToDevice);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

    //*sigma_ array
  err = cudaMemcpy(this->sigma, utils::convertPnlVectToFloat(bs->sigma_->array,bs->size_), bs->sigma_->size*sizeof(float), cudaMemcpyHostToDevice);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

    //*spot_ array
  err = cudaMemcpy(this->spot, utils::convertPnlVectToFloat(bs->spot_->array,bs->size_), bs->spot_->size*sizeof(float), cudaMemcpyHostToDevice);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

    //*chol_ array
  err = cudaMemcpy(this->chol, utils::convertPnlVectToFloat(bs->chol->array,bs->chol->m*bs->chol->n), bs->chol->m*bs->chol->n*sizeof(float), cudaMemcpyHostToDevice);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }
}

void CudaLib::memcpyMonteCarlo(MonteCarlo* mc){
    //Chargement en mémoire de l'option
  memcpyOption(mc->opt_);
    //Chargement en mémoire du modèle de Black&Scholes
  memcpyBS(mc->mod_);  
}




