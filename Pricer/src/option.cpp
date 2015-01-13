#include <iostream>

using namespace std;

#include "option.h"


Option::Option(const double T_, const int timeSteps_, const int size_){
  this->T_ = T_;
  this->TimeSteps_ = timeSteps_;
  this->size_ = size_;
}

Option::~Option(){}

