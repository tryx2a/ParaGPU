/*
 * Méthodes dédiées à être exécuté sur le noyau.
 *
 */

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
    scalarResult += chol[i + assetIndex*sizeOpt]*generate(globalState, maxDevice);
  }

  float sigmaAsset = sigma[assetIndex]; // recup la vol de l'asset

  float expArg = sqrtf(h)*scalarResult*sigmaAsset + h*(r - (sigmaAsset*sigmaAsset/2));
  float fexp = expf(expArg);

  return fexp;
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
      path[ind + j + i*sizeOpt] = computeIterationGPU( path[ind + j + (i-1)*sizeOpt], T/N, j, globalState, sizeOpt, chol, sigma, r, maxDevice) * path[ind + j + (i-1)*sizeOpt];
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
    float payOff = payoffBasketGPU(tabPath, payoffCoeff, timeSteps, strike, size, T, maxDevice);

    tabPrice[ind] = payOff;
    tabIC[ind] = payOff * payOff;

}