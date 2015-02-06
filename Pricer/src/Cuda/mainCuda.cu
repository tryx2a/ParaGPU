#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cmath>

#include "../parser.h"
#include "../Method/mc.h"
#include "cudaLib.h"
#include "pricer_kernel.cuh"

#include <curand.h>
#include <curand_kernel.h>



int main(int argc, char ** argv) {

	const char *infile = argv[1];

	Param *P = new Parser(infile);
  MonteCarlo *mc = new MonteCarlo(P);

  //Creation du CudaLib
  CudaLib* cudaL = new CudaLib(mc);

  //Alloc des etats dans le GPU
  curandState* devStates;
  cudaMalloc ( &devStates, (mc->samples_)*sizeof( curandState ) );
  
  /// Initialise la grille et les dimensions de chaque bloc
  dim3 DimGrid(mc->samples_/cudaL->maxDevice,1,1);
  dim3 DimBlock(cudaL->maxDevice,1,1);

  float strike = (dynamic_cast<OptionBasket *>(mc->opt_))->strike_;

  //Initialisation du noyau
  priceGPU <<<DimGrid, DimBlock>>>(cudaL->tabPrice, cudaL->tabIC, cudaL->tabPath, mc->mod_->size_, mc->mod_->r_, cudaL->spot, cudaL->sigma, cudaL->chol,
                                    mc->opt_->T_, mc->opt_->TimeSteps_, cudaL->payoffCoeff, strike, devStates, cudaL->maxDevice, unsigned(time(NULL)));

  
  float *priceTable = new float[mc->samples_];
  float *icTable = new float[mc->samples_];


  cudaError_t err; 

  err = cudaMemcpy(priceTable, cudaL->tabPrice, mc->samples_*sizeof(float), cudaMemcpyDeviceToHost);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(icTable, cudaL->tabIC, mc->samples_*sizeof(float), cudaMemcpyDeviceToHost);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

 
  float prixReduction = 0.0;
  float varianceReduction = 0.0;
  float coeffActu = exp(-mc->mod_->r_*mc->opt_->T_);

  for(int i = 0; i<mc->samples_; i++){
      prixReduction += priceTable[i];
      varianceReduction += icTable[i];
  }

  prixReduction /= mc->samples_;
  varianceReduction /= mc->samples_;
  
  float varEstimator = exp(- 2 * (mc->mod_->r_ * mc->opt_->T_)) * (varianceReduction - (prixReduction*prixReduction));

  float prixFin = prixReduction*coeffActu;
  float ic = 2 * 1.96 * sqrt(varEstimator)/sqrt(mc->samples_);

  std::cout<<"Prix : "<<prixFin<<std::endl;
  std::cout<<"IC : "<<ic<<std::endl;


  free(priceTable);
  free(icTable);

  delete P;
  delete mc;

 
	return 0;
}
