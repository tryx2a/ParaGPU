#include <iostream>

using namespace std;

#include "optionAsian.h"

OptionAsian::OptionAsian(double T_, int TimeSteps_, int size_, double strike_):Option(T_,TimeSteps_,size_)
{
	this->strike_=strike_;
}


OptionAsian::~OptionAsian(){}

double OptionAsian::payoff(const PnlMat *path)
{
	// Put the first column of path in payoffCoeff_
	PnlVect *p= pnl_vect_new();
	pnl_mat_get_col(p,path,0);
	double sum_flow = 0.0;
	for (int i=0;i<this->TimeSteps_+1;i++){
		sum_flow+=GET(p,i);
	}
	pnl_vect_free(&p);
	double payoff = (sum_flow/(this->TimeSteps_ + 1))-(this->strike_);
	
	if (payoff<0.0) {
		return 0.0;
	}else{
		return payoff;
	}
}
