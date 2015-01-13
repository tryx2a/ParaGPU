#ifndef _MC_H
#define _MC_H

#include "optionBasket.h"
#include "optionAsian.h"
#include "optionBarrierLow.h"
#include "optionBarrierUp.h"
#include "optionBarrier.h"
#include "optionPerformance.h"
#include "bs.h"
#include "pnl/pnl_random.h"
#include "parser.h"

class MonteCarlo
{
public:
  BS *mod_; /*! pointeur vers le mod�le */
  Option *opt_; /*! pointeur sur l'option */
  PnlRng *rng; /*! pointeur sur le g�n�rateur */
  double h_; /*! pas de diff�rence finie */
  int H_; /* nombre de p�riode de rebalancement*/
  int samples_; /*! nombre de tirages Monte Carlo */

  MonteCarlo(Param* P);

  ~MonteCarlo();

  /**
   * Calcule le prix de l'option � la date 0
   *
   * @param[out] prix valeur de l'estimateur Monte Carlo
   * @param[out] ic largeur de l'intervalle de confiance
   */
  void price(double &prix, double &ic);

  /**
   * Calcule le prix de l'option � la date t
   *
   * @param[in]  past contient la trajectoire du sous-jacent
   * jusqu'� l'instant t
   * @param[in] t date � laquelle le calcul est fait
   * @param[out] prix contient le prix
   * @param[out] ic contient la largeur de l'intervalle
   * de confiance sur le calcul du prix
   */
  void price(const PnlMat *past, double t, double &prix, double &ic);

  /**
   * Calcule le delta de l'option � la date t
   *
   * @param[in] past contient la trajectoire du sous-jacent
   * jusqu'� l'instant t
   * @param[in] t date � laquelle le calcul est fait
   * @param[out] delta contient le vecteur de delta
   * @param[out] ic contient la largeur de l'intervalle
   * de confiance sur le calcul du delta
   */
  void delta(const PnlMat *past, double t, PnlVect *delta, PnlVect *ic);

  /**
   * Cette m�thode cr��e et retourne la bonne instance d'option
   * en fonction de la key pass�e en param�tre.
   *
   * @param[in] key contient le type de l'option
   * @param[in] P contient les donn�es n�cessaire pour 
   * la cr�ation de l'option
   * @param[out] retourne la bonne instance d'option
   */
  static Option* createOption(Param *P);
  
  void freeRiskInvestedPart(PnlVect *V,double T, double &profitLoss);

};

#endif /* _MC_H */

