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
  
  //Creation du CudaLib permettant d'allouer la mémoire nécessaire sur le GPU et faisant les copies
  CudaLib* cudaL = new CudaLib(mc);

  //Alloc des états dans le GPU
  curandState* devStates;
  cudaMalloc ( &devStates, (mc->samples_)*sizeof( curandState ) );
  
  /// Initialise la grille et les dimensions de chaque bloc
  dim3 DimGrid(mc->samples_/cudaL->maxDevice,1,1);
  dim3 DimBlock(cudaL->maxDevice,1,1);
  
  //Appel du noyau
  priceGPU <<<DimGrid, DimBlock>>>(cudaL->tabPrice, cudaL->tabVar, cudaL->tabPath, mc->mod_->size_, mc->mod_->r_, cudaL->spot, cudaL->sigma, cudaL->chol,
                                    mc->opt_->T_, mc->opt_->TimeSteps_, cudaL->payoffCoeff, cudaL->lowerBarrier, cudaL->upperBarrier, cudaL->strike, 
                                    mc->opt_->id_, devStates, cudaL->maxDevice, unsigned(time(NULL)));

  
  float *priceTable = new float[mc->samples_];
  float *varTable = new float[mc->samples_];

  cudaError_t err; 

  //Copie du tableau de prix calculé sur le GPU sur le processeur
  err = cudaMemcpy(priceTable, cudaL->tabPrice, mc->samples_*sizeof(float), cudaMemcpyDeviceToHost);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

  //Copie du tableau de variance calculé sur le GPU sur le processeur
  err = cudaMemcpy(varTable, cudaL->tabVar, mc->samples_*sizeof(float), cudaMemcpyDeviceToHost);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }

  /*
   * Réduction des calculs et calcul du prix et de l'IC
   */
  float prixReduction = 0.0;
  float varianceReduction = 0.0;
  float coeffActu = exp(-mc->mod_->r_*mc->opt_->T_);

  for(int i = 0; i<mc->samples_; i++){
      prixReduction += priceTable[i];
      varianceReduction += varTable[i];
  }

  prixReduction /= mc->samples_;
  varianceReduction /= mc->samples_;
  
  float varEstimator = exp(- 2 * (mc->mod_->r_ * mc->opt_->T_)) * (varianceReduction - (prixReduction*prixReduction));

  float prixFin = prixReduction*coeffActu;
  float ic = 2 * 1.96 * sqrt(varEstimator)/sqrt(mc->samples_);

  std::cout<<"Prix : "<<prixFin<<std::endl;
  std::cout<<"IC : "<<ic<<std::endl;


  free(priceTable);
  free(varTable);

  delete P;
  delete mc;

 
	return 0;
}
