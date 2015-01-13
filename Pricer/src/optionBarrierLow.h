#ifndef _OPTIONBARRIERLOW_H
#define _OPTIONBARRIERLOW_H

#include "option.h"

class OptionBarrierLow : public Option
{
public:
  double strike_;
  PnlVect* payoffCoeff_;
  PnlVect* lowerBarrier_;

  OptionBarrierLow(const double T_, const int timeSteps_, const int size_, const double strike_, const PnlVect* payoffCoeff_, const PnlVect* lowerBarrier_);

  double payoff(const PnlMat *path);

  ~OptionBarrierLow();
};

#endif //_OPTIONBARRIERLOW_H
