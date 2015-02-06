#ifndef _OPTIONBARRIERUP_H
#define _OPTIONBARRIERUP_H

#include "option.h"

class OptionBarrierUp : public Option
{
public:
  double strike_;
  PnlVect* payoffCoeff_;
  PnlVect* upperBarrier_;

  OptionBarrierUp(const double T_, const int timeSteps_, const int size_, const double strike_, const PnlVect* payoffCoeff_, const PnlVect* upperBarrier_);

  double payoff(const PnlMat *path);

  ~OptionBarrierUp();
};

#endif //_OPTIONBARRIERUP_H
