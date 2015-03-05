#include "../../src//Model/bs.h"
#include "../../src/parser.h"
#include "../../src/Method/mc.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>

using namespace std;

int main(int argc, char const *argv[])
{
  const char *infile = argv[1];

  Param *P = new Parser(infile);
  MonteCarlo *mc = new MonteCarlo(P);
  double prix;
  double ic;
  mc->price(prix, ic);
  cout << "Output : " << endl << "Price : " << prix << endl << "Confidence interval : " << ic << endl;

  delete P;
  delete mc;
  return 0;
}
