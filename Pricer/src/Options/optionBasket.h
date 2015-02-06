#ifndef _OPTIONBASKET_H
#define _OPTIONBASKET_H

#include "option.h"

class OptionBasket : public Option
{
public:
  double strike_;
  PnlVect * payoffCoeff_;

  OptionBasket(const double T_, const int timeSteps_, const int size_, const double strike_, const PnlVect* payoffCoeff);
  
  ~OptionBasket();

  double payoff(const PnlMat *path);
};

#endif //_OPTIONBASKET_H
