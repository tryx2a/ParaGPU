#include "../../src/bs.h"
#include "../../src/parser.h"
#include "../../src/mc.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>

using namespace std;

int main(int argc, char const *argv[])
{
  const char *infile = argv[1];

  Param *P = new Parser(infile);
  MonteCarlo *mc = new MonteCarlo(P);

  PnlVect *delta = pnl_vect_create(40);

  mc->delta(NULL,0,delta,NULL);

  cout << "Delta :" << endl;
  pnl_vect_print(delta);

  delete P;
  delete mc;
  return 0;
}
