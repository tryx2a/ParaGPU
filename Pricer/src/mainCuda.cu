//#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "parser.h"
#include "mc.h"
#include "utils.h"

#include "cudaLib.h"
#include <curand.h>
#include <curand_kernel.h>

//Generateur aleatoire
__device__ float generate( curandState* globalState, int maxDevice) 
{
    int ind = maxDevice * blockIdx.x + threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_normal( &localState );
    globalState[ind] = localState;
    return RANDOM;
}


//Generateur aleatoire
__device__ void setup_kernel ( curandState * state, unsigned long seed, int maxDevice)
{
    int ind = maxDevice * blockIdx.x + threadIdx.x;
    curand_init ( seed, ind, 0, &state[ind] );
}



__device__ float computeIterationGPU(float currentPrice, float h, int assetIndex, curandState* globalState, int sizeOpt, float* chol, float* sigma, float r, int maxDevice){
  float scalarResult = 0.0;

  //Produit scalaire
  for(int i=0; i<sizeOpt;i++){
    scalarResult += chol[i + assetIndex*sizeOpt]*generate(globalState, maxDevice); //vectGauss[i];
  }

  float sigmaAsset = sigma[assetIndex]; // recup la vol de l'asset

  float expArg = sqrt(h)*scalarResult*sigmaAsset + h*(r - (sigmaAsset*sigmaAsset/2));

  return currentPrice*exp(expArg);
}



__device__ void assetGPU(float* path, float T, int N, int sizeOpt, float* spot, float* sigma, float* chol, float r, curandState* globalState, int maxDevice){

  int ind = (maxDevice * blockIdx.x + threadIdx.x) * (sizeOpt * (N+1));

  //Recopie des spots
  for(int j=0; j < sizeOpt ; j++){
    path[ind + j] = spot[j];
  }

  //Calcul de trajectoire
  for(int i = 1 ; i < N+1 ; i++){

    for(int j = 0 ; j < sizeOpt ; j ++){
      path[ind + j + i*sizeOpt] = computeIterationGPU( path[ind + j + (i-1)*sizeOpt], T/N, j, globalState, sizeOpt, chol, sigma, r, maxDevice);
    }

  }


}


__device__ float payoffBasketGPU(float* path, float* payoffCoeff, int timeSteps, float strike, int sizeOpt, float T, int maxDevice){ //m ligne, n colonnes

  int ind = (maxDevice * blockIdx.x + threadIdx.x) * (sizeOpt * (timeSteps+1));

  float res = 0.0;
  int indiceDernierligne = timeSteps*sizeOpt;
  for(int j = 0; j<sizeOpt; j++){
      res += path[ind + j + indiceDernierligne] * payoffCoeff[j];
  }
  res -= strike;
  if(res<=0.0){
    res = 0.0;
  }
  return res;
}



__global__ void priceGPU(float *tabPrice, float *tabIC, float *tabPath, int size, float r, 
                         float *spot, float *sigma, float *chol, float T, int timeSteps, float *payoffCoeff,
                         float strike, curandState* globalState, int maxDevice, unsigned long seed) {

    int ind = maxDevice * blockIdx.x + threadIdx.x;
    setup_kernel (globalState, seed, maxDevice);
    
    assetGPU(tabPath, T, timeSteps, size, spot, sigma, chol, r, globalState, maxDevice);
    //float payOff = payoffBasketGPU(tabPath, payoffCoeff, timeSteps, strike, size, T, maxDevice);

    //tabPrice[ind] = payOff;
    tabPrice[ind] = 3.14;//tabPath[0];
    //tabIC[ind] = payOff * payOff;

}



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


  cudaError_t err; 

  err = cudaMemcpy(priceTable, cudaL->tabPrice, mc->samples_*sizeof(float), cudaMemcpyDeviceToHost);
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
    exit(EXIT_FAILURE);
  }


  //float *icTable = new float[mc->samples_]; 

  for(int i = 0; i<10; i++){
    std::cout<<priceTable[i]<<std::endl;
  }


  delete P;
  delete mc;
  delete priceTable;

 
	return 0;
}
