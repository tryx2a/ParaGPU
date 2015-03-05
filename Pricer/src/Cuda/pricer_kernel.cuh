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


//Initialisation du generateur aleatoire
__device__ void setup_kernel ( curandState * state, unsigned long seed, int maxDevice)
{
    int ind = maxDevice * blockIdx.x + threadIdx.x;
    curand_init ( seed, ind, 0, &state[ind] );
}


//Méthode permettant de calculer le prix d'un actif en t+1 connaissant sa valeur en t
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


//Méthode remplissant une matrice path avec les trajectoires des sous jacents de l'option à pricer
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


//Calcul du payOff pour une optionBasket
__device__ float payoffBasketGPU(float* path, float* payoffCoeff, int timeSteps, float strike, int sizeOpt, float T, int maxDevice){ 

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


//Calcul du payOff pour une optionAsian
__device__ float payoffAsianGPU(float* path, int timeSteps, float strike, int sizeOpt, float T, int maxDevice){ 

  int ind = (maxDevice * blockIdx.x + threadIdx.x) * (sizeOpt * (timeSteps+1));

  float sum_flow = 0.0;

  for (int i = 0; i<timeSteps+1; i++){
    sum_flow += path[ind + sizeOpt*i];
  }

  float payoff = (sum_flow/(timeSteps + 1)) - strike;
  
  if (payoff<0.0) {
    return 0.0;
  }else{
    return payoff;
  }

}


//Calcul du payoff d'une optionBarrierLow
__device__ float payoffBarrierLowGPU(float* path, float* payoffCoeff, float* lowerBarrier, int timeSteps, float strike, int sizeOpt, float T, int maxDevice){

  int ind = (maxDevice * blockIdx.x + threadIdx.x) * (sizeOpt * (timeSteps+1));

  for (int i = 0; i < timeSteps+1; i++){
    for (int j = 0; j < sizeOpt; j++){
        if (path[ind + j + i*sizeOpt] < lowerBarrier[j]){
          return 0;
        }
    }
  }

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


//Calcul du payoff d'une optionBarrierUp
__device__ float payoffBarrierUpGPU(float* path, float* payoffCoeff, float* upperBarrier, int timeSteps, float strike, int sizeOpt, float T, int maxDevice){

  int ind = (maxDevice * blockIdx.x + threadIdx.x) * (sizeOpt * (timeSteps+1));

  for (int i = 0; i < timeSteps+1; i++){
    for (int j = 0; j < sizeOpt; j++){
        if (path[ind + j + i*sizeOpt] > upperBarrier[j]){
          return 0;
        }
    }
  }

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


//Calcul du payoff d'une optionBarrier
__device__ float payoffBarrierGPU(float* path, float* payoffCoeff, float* upperBarrier, float* lowerBarrier, int timeSteps, float strike, int sizeOpt, float T, int maxDevice){

  int ind = (maxDevice * blockIdx.x + threadIdx.x) * (sizeOpt * (timeSteps+1));

  for (int i = 0; i < timeSteps+1; i++){
    for (int j = 0; j < sizeOpt; j++){
        if (path[ind + j + i*sizeOpt] > upperBarrier[j] || path[ind + j + i*sizeOpt] < lowerBarrier[j]){
          return 0;
        }
    }
  }

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


//Calcul du payoff d'une optionPerformance
__device__ float payoffPerformance(float* path, float* payoffCoeff, int timeSteps, int sizeOpt, float T, int maxDevice){

  int ind = (maxDevice * blockIdx.x + threadIdx.x) * (sizeOpt * (timeSteps+1));

  float ratio = 0.0;
  float num = 0.0; 
  float den = 0.0;

  //Sum coeff ratio
  for(int i = 1 ; i <= timeSteps ; i++){
  
    //Compute numerator and denom
    for(int d = 0 ; d < sizeOpt ; d++){
      
      num += path[ind + d + i*sizeOpt] * payoffCoeff[d];
      den += path[ind + d + (i-1)*sizeOpt] * payoffCoeff[d];
      
    }
    //Compute of ratio
    ratio += num/den; 
  }

  ratio = ratio/timeSteps - 1;
  
  //Refresh ratio
  if(ratio < 0){
    ratio = 0;
  }

  //Compute min
  if(ratio > 0.1){
    return 1.1;
  }else{
    return 1 + ratio;
  }  

}


//Choix de la bonne méthode de payoff à appeler
__device__ float payoffGPU(float* path, float* payoffCoeff, float* upperBarrier, float* lowerBarrier, 
                          int timeSteps, float strike, int sizeOpt, float T, int maxDevice, int idOpt) {

  if(idOpt == 1){ //Option Asian
    return payoffAsianGPU(path, timeSteps, strike, sizeOpt, T, maxDevice);
  }
  else if(idOpt == 2){ //Option Barrière
    return payoffBarrierGPU(path, payoffCoeff, upperBarrier, lowerBarrier, timeSteps, strike, sizeOpt, T, maxDevice);
  }
  else if(idOpt == 3){ //Option Barrière Basse
    return payoffBarrierLowGPU(path, payoffCoeff, lowerBarrier, timeSteps, strike, sizeOpt, T, maxDevice);
  }
  else if(idOpt == 4){ //Option Barrière Haute
    return payoffBarrierUpGPU(path, payoffCoeff, upperBarrier, timeSteps, strike, sizeOpt, T, maxDevice);
  }
  else if(idOpt == 5){ //Option Basket
    return payoffBasketGPU(path, payoffCoeff, timeSteps, strike, sizeOpt, T, maxDevice);
  }
  else if(idOpt == 6){ //Option Performance
    return payoffPerformance(path, payoffCoeff, timeSteps, sizeOpt, T, maxDevice);
  }
  else{ //Ne devrait jamais arriver
    return 0.0;
  }
} 


//Calcul du prix d'une option donnée
__global__ void priceGPU(float *tabPrice, float *tabVar, float *tabPath, int size, float r, 
                         float *spot, float *sigma, float *chol, float T, int timeSteps, float *payoffCoeff,
                         float *lowerBarrier, float *upperBarrier, float strike, int idOption,
                         curandState* globalState, int maxDevice, unsigned long seed) {

    int ind = maxDevice * blockIdx.x + threadIdx.x;
    setup_kernel (globalState, seed, maxDevice);
    
    assetGPU(tabPath, T, timeSteps, size, spot, sigma, chol, r, globalState, maxDevice);
    float payOff = payoffGPU(tabPath, payoffCoeff, upperBarrier, lowerBarrier, timeSteps, strike, size, T, maxDevice, idOption);

    tabPrice[ind] = payOff;
    tabVar[ind] = payOff * payOff;
}


//Effectue une réduction d'un tableau avec une puissance de 2 éléments
__global__ void block_sum(const float *input,
                          float *per_block_results,
                          const size_t n)
{
  extern __shared__ float sdata[];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // load input into __shared__ memory
  float x = 0;
  if(i < n)
  {
    x = input[i];
  }
  sdata[threadIdx.x] = x;
  __syncthreads();

  // contiguous range pattern
  for(int offset = blockDim.x / 2;
      offset > 0;
      offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      // add a partial sum upstream to our own
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    }

    // wait until all threads in the block have
    // updated their partial sums
    __syncthreads();
  }

  // thread 0 writes the final result
  if(threadIdx.x == 0)
  {
    per_block_results[blockIdx.x] = sdata[0];
  }
}




