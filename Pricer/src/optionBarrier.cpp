#include "optionBarrier.h"
#include "utils.h"
#include <assert.h> 
#include <iostream>

using namespace std;

OptionBarrier::OptionBarrier(const double T_, const int timeSteps_, const int size_, const double strike_, const PnlVect* payoffCoeff_, const PnlVect* lowerBarrier_, const PnlVect* upperBarrier_) : Option(T_, timeSteps_, size_)
{
  this->strike_ = strike_;
  this->payoffCoeff_ = pnl_vect_copy(payoffCoeff_);
  this->lowerBarrier_ = pnl_vect_copy(lowerBarrier_);
  this->upperBarrier_ = pnl_vect_copy(upperBarrier_);
}

OptionBarrier::~OptionBarrier(){
  if (&(this->payoffCoeff_) != NULL)
    pnl_vect_free(&(this->payoffCoeff_));
  if (&(this->lowerBarrier_) != NULL)
    pnl_vect_free(&(this->lowerBarrier_));
  if (&(this->upperBarrier_) != NULL)
    pnl_vect_free(&(this->upperBarrier_));
}

double OptionBarrier::payoff(const PnlMat *path)
{
  assert(lowerBarrier_->size == path->n);
  for (int i = 0; i < path->m; ++i)
  {
    for (int d = 0; d < path->n; ++d)
    {
      if (MGET(path, i, d) < GET(this->lowerBarrier_, d) || GET(this->upperBarrier_, d) < MGET(path, i, d) ){
        return 0;
      }
    }
  }

  return utils::computePayoff(path, this->payoffCoeff_, this->TimeSteps_, this->strike_);
}
