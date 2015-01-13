#include "optionBarrierUp.h"
#include "utils.h"
#include <iostream>
#include "assert.h"

using namespace std;

OptionBarrierUp::OptionBarrierUp(const double T_, const int timeSteps_, const int size_, const double strike_, const PnlVect* payoffCoeff_, const PnlVect* upperBarrier_) : Option(T_, timeSteps_, size_)
{
  this->strike_ = strike_;
  this->payoffCoeff_ = pnl_vect_copy(payoffCoeff_);
  this->upperBarrier_ = pnl_vect_copy(upperBarrier_);
}

OptionBarrierUp::~OptionBarrierUp(){
  if (&(this->payoffCoeff_) != NULL)
    pnl_vect_free(&(this->payoffCoeff_));
  if (&(this->upperBarrier_) != NULL)
    pnl_vect_free(&(this->upperBarrier_));
}

double OptionBarrierUp::payoff(const PnlMat *path)
{
  assert(upperBarrier_->size == path->n);
  for (int i = 0; i < path->m; ++i)
  {
    for (int d = 0; d < path->n; ++d)
    {
      if (GET(this->upperBarrier_, d) < MGET(path, i, d)){
        return 0;
      }
    }
  }

  return utils::computePayoff(path, this->payoffCoeff_, this->TimeSteps_, this->strike_);
}
