#ifndef _OPTIONBARRIER_H
#define _OPTIONBARRIER_H

#include "option.h"

class OptionBarrier : public Option
{
public:
  double strike_;
  PnlVect* payoffCoeff_;
  PnlVect* lowerBarrier_;
  PnlVect* upperBarrier_;

  OptionBarrier(const double T_, const int timeSteps_, const int size_, const double strike_, const PnlVect* payoffCoeff_, const PnlVect* lowerBarrier_, const PnlVect* upperBarrier_);

  double payoff(const PnlMat *path);

  ~OptionBarrier();
};

#endif //_OPTIONBARRIER_H
