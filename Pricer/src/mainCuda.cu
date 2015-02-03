//#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "parser.h"
#include "mc.h"
#include "cudaLib.h"
#include "utils.h"

//Generateur aleatoire
__device__ float generate( curandState* globalState, int ind ) 
{
    int ind = threadIdx.x;
    curandState localState = globalState[ind];
    //float RANDOM = curand_uniform( &localState );
    float RANDOM = curand_normal( &localState );
    globalState[ind] = localState;
    return RANDOM;
}
//Generateur aleatoire
__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

__device__ float computeIterationGPU(float currentPrice, float h, int assetIndex, float* vectGauss, int sizeOpt, float* chol, float* sigma, float r){
  float scalarResult = 0.0;

  //Produit scalaire
  for(int i=0; i<sizeOpt;i++){
    scalarResult += chol[i + assetIndex*sizeOpt]*vectGauss[i];
  }

  float sigmaAsset = sigma[assetIndex]; // recup la vol de l'asset

  float expArg = sqrt(h)*scalarResult*sigmaAsset + h*(r - (sigmaAsset*sigmaAsset/2));

  return currentPrice*exp(expArg);
}

__device__ void assetGPU(float* path, int m, int n, float T, int N, int sizeOpt, float* spot, float* sigma, float* chol, float r){

  //Recopie des spots
  for(int j=0; j < sizeOpt ; j++){
    path[j] = spot[j];
  }

  //Creation vecteur gausse
  float* vectGauss = new float[sizeOpt];

  //Calcul de trajectoire
  for(int i =1 ; i < N+1 ; i++){
    //invocation du vecteur gaussien
    

    for(int j = 0 ; j < sizeOpt ; j ++){
      path[j + i*sizeOpt] = computeIterationGPU( path[j + (i-1)*sizeOpt], T/N, j, vectGauss, sizeOpt, chol, sigma, r);
    }

  }

}


__device__ float payoffBasketGPU(float* path, float* payoffCoeff, int timeSteps, float strike, int sizeOpt, float T){ //m ligne, n colonnes

  float res = 0.0;
  int indiceDernierligne = timeSteps*sizeOpt;
  for(int j = 0; j<sizeOpt; j++){
      res += path[j + indiceDernierligne] * payoffCoeff[j];
  }
  res -= strike;
  if(res<=0.0){
    res = 0.0;
  }
  return res;
}

__global__ void priceGPU(float &prix, float &ic, float h, float H, int samples, int size, float r, float *trend,
                        float rho, float *spot, float *sigma, float *chol, float T, int timeSteps, float *payoffCoeff, float strike) {

    //int col = blockIdx.x * blockDim.x + threadIdx.x; 
    //int row = blockIdx.y * blockDim.y + threadIdx.y; 

}

// __global__ void matrixMultiply(float * A, float * B, float * C,
//              int numARows, int numAColumns,
//              int numBRows, int numBColumns,
//              int numCRows, int numCColumns) {
//     /// Insérer le code
//     int Row = blockIdx.y * blockDim.y + threadIdx.y;
//     int Col = blockIdx.x * blockDim.x + threadIdx.x;

//     if((Row < numARows) && (Col < numBColumns)){
//       float Pvalue = 0.0;
//       for(int k = 0; k < numAColumns; ++k){
//         Pvalue += A[Row*numAColumns+k]*B[k*numBColumns+Col];
//       }
  
//       C[Row*numCColumns+Col] = Pvalue;
//     }
    
// }





int main(int argc, char ** argv) {

	const char *infile = argv[1];

	Param *P = new Parser(infile);
  MonteCarlo *mc = new MonteCarlo(P);
  	//double prix;
  	//double ic;

  	//mc->price(prix,ic);
  	//std::cout << "Prix  : "<< prix << std::endl;
  	//std::cout << "IC : "<< ic << std::endl;

  CudaLib* cudaL = new CudaLib(mc);
  	//Allocation des différents paramètres dans le GPU
  	//cudaL->allocMonteCarlo(mc);

    float* test = utils::convertPnlVectToFloat(mc->mod_->spot_->array, mc->mod_->size_);

    for(int i=0; i< mc->mod_->size_; i++){
      std::cout << "Component : "<<test[i]<<std::endl;
    }

  	delete P;
  	delete mc;


 
	return 0;
}
