#include <iostream>

using namespace std;

#include "optionBasket.h"
#include "utils.h"

/**
 * Constructeur de la classe OptionBasket
 */
OptionBasket::OptionBasket(const double T_, const int timeSteps_, const int size_, const double strike_, const PnlVect* payoffCoeff_) : Option(T_, timeSteps_, size_){
  this->strike_ = strike_;
  this->payoffCoeff_ = pnl_vect_copy(payoffCoeff_);
}

/**
 * Destructeur de la classe OptionBasket
 */
OptionBasket::~OptionBasket(){
  if (&(this->payoffCoeff_) != NULL)
    pnl_vect_free(&(this->payoffCoeff_));
}

/**
 * Implémentation de la méthode de payoff
 */
double OptionBasket::payoff(const PnlMat *path){
  return utils::computePayoff(path, this->payoffCoeff_, this->TimeSteps_, this->strike_);
}
