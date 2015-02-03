//#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "parser.h"
#include "mc.h"
#include "cudaLib.h"
#include "utils.h"



__device__ void assetGPU(float* path, int m, int n, float T, int N){

}

__device__ float payoffBasketGPU(float* path, float* payoffCoeff, int timeSteps, float strike, int sizeOpt, float T){ //m ligne, n colonnes

  float res = 0.0;
  
  int indiceDernierligne = timeSteps*sizeOpt;

  for(int j = 0; j<sizeOpt; j++){
      res += path[j + indiceDernierligne] * payoffCoeff[j];
  }

  res -= strike;

  if(res<=0.0){
    return 0.0;
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
