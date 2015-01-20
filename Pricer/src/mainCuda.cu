//#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "parser.h"
#include "mc.h"
#include "cudaLib.h"

int main(int argc, char ** argv) {

	const char *infile = argv[1];

	Param *P = new Parser(infile);
  	MonteCarlo *mc = new MonteCarlo(P);
  	//double prix;
  	//double ic;

  	//mc->price(prix,ic);
  	//std::cout << "Prix  : "<< prix << std::endl;
  	//std::cout << "IC : "<< ic << std::endl;

  	//Permet de determiner
  	CudaLib* cudaL = new CudaLib();
  	//512 est dans le maxDevice



  	delete P;
  	delete mc;
 
	return 0;
}