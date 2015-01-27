//#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "parser.h"
#include "mc.h"
#include "cudaLib.h"
#include "utils.h"



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

    float* test = utils::convertPnlVectToFloat(mc->mod_->spot_);

    for(int i=0; i< mc->mod_->size_; i++){
      std::cout << "Component : "<<test[i]<<std::endl;
    }

  	delete P;
  	delete mc;
 
	return 0;
}
