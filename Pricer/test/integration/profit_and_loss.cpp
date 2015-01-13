#include <iostream>
#include "../../src/bs.h"
#include "../../src/parser.h"
#include "../../src/mc.h"
using namespace std;

int main(int argc, char **argv)
{

  double T;
  int H;
  char *infile = argv[1];
  Param *P = new Parser(infile);
  
  
  P->extract("maturity", T);
  P->extract("hedging dates number", H);

  MonteCarlo *mc;
  mc = new MonteCarlo(P);

  PnlVect *V;
  V = pnl_vect_create(H+1);
  double profitLoss;
  mc->freeRiskInvestedPart(V,T,profitLoss);
  cout << "Price vector V: "<<endl;
  pnl_vect_print(V);
  cout << "Profit and Loss: "<< profitLoss<<endl;
  
  pnl_vect_free(&V);
  exit(0);
}
