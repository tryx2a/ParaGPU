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
#include <sys/time.h>



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


  /*
   * Réduction
   */ 

  //Constantes définissant la grille à utiliser pour l'allocation de la grille
  int num_elements = mc->samples_;
  size_t block_size = cudaL->maxDevice;
  size_t num_blocks = mc->samples_/cudaL->maxDevice;

  //Allocation des variables qui contiendront les résultats des réductions
  float *d_partial_sums_and_total_price;
  float *device_result_price;
  float *d_partial_sums_and_total_var;
  float *device_result_var;
  cudaMalloc((void**)&d_partial_sums_and_total_price, sizeof(float) * num_blocks);
  cudaMalloc((void**)&device_result_price, sizeof(float));
  cudaMalloc((void**)&d_partial_sums_and_total_var, sizeof(float) * num_blocks);
  cudaMalloc((void**)&device_result_var, sizeof(float));

  float payoffReduction = 0.0;
  float payoffSquareReduction = 0.0;
  float host_result_price = 0.0;
  float host_result_var = 0.0;

  int puissance = (int)(log(num_elements)/log(2));
  float *host_tab_price = new float[num_blocks];
  float *host_tab_var = new float[num_blocks];

  while( puissance >= 9){ // car 2^9 = 512

    // launch one kernel to compute, per-block, a partial sum
    block_sum<<<num_blocks,block_size,block_size * sizeof(float)>>>(cudaL->tabPrice + (mc->samples_ - num_elements), d_partial_sums_and_total_price, num_elements);
    block_sum<<<num_blocks,block_size,block_size * sizeof(float)>>>(cudaL->tabVar + (mc->samples_ - num_elements), d_partial_sums_and_total_var, num_elements);

    int blocks = (int)(pow(2.0,puissance))/cudaL->maxDevice;

    cudaMemcpy(host_tab_price, d_partial_sums_and_total_price, sizeof(float)*blocks, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_tab_var, d_partial_sums_and_total_var, sizeof(float)*blocks, cudaMemcpyDeviceToHost);

    // copy the result back to the host
    host_result_price = 0.0;
    host_result_var = 0.0;

    for(int i = 0; i<blocks; i++){
      host_result_price += host_tab_price[i]; 
      host_result_var += host_tab_var[i];
    }

    payoffReduction += host_result_price;
    payoffSquareReduction += host_result_var;

    num_elements -= (int)(pow(2.0,puissance));
    num_blocks = num_elements/cudaL->maxDevice;
    puissance = (int)(log(num_elements)/log(2));

  }


  payoffReduction /= mc->samples_;
  payoffSquareReduction /= mc->samples_;

  float coeffActu = exp(-mc->mod_->r_*mc->opt_->T_);
  float varEstimator = exp(- 2 * (mc->mod_->r_ * mc->opt_->T_)) * (payoffSquareReduction - (payoffReduction*payoffReduction));

  float prixFin = payoffReduction*coeffActu;
  float ic = 2 * 1.96 * sqrt(varEstimator)/sqrt(mc->samples_);

  std::cout<<"Prix : "<<prixFin<<std::endl;
  std::cout<<"IC : "<<ic<<std::endl;


  // deallocate device memory
  cudaFree(d_partial_sums_and_total_price);
  cudaFree(device_result_price);
  cudaFree(d_partial_sums_and_total_var);
  cudaFree(device_result_var);
  delete host_tab_price;
  delete host_tab_var;
  delete P;
  delete mc;
  delete cudaL;
 
	return 0;
}

