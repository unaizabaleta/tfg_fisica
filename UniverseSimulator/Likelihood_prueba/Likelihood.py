#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 21:59:59 2022

@author: jesus
"""

from scipy import stats
import statsmodels.api as sm
import numpy as np
from statsmodels.base.model import GenericLikelihoodModel
from prueba_likelihood import simulator


class MyLikelihood(GenericLikelihoodModel):
    
    def __init__(self, endog, exog, fixed_params, **kwds):
        self.fixed_params = fixed_params
        print(self.fixed_params)
        super(MyLikelihood, self).__init__(endog, exog, **kwds)



    def loglike(self, params):
        X = self.exog #En X metemos todos los datos necesarios para inicializar el Universo

        Y = self.endog #En Y metemos los datos de los 10 años simulados (¿o solo la fecha final?)
        #Aquí metemos que simule los 10 años y que genere todos los Y_ de cada año para ese municipio
        params = np.append(self.fixed_params,params)
        temp = simulator(np.array(X) ,np.array(params))
        # Una vez tenemos eso sacamos el error medio cuadrático
        res = np.sum((temp.evaluation() - np.array(Y))**2)
        return -res


if __name__ == "__main__":
    a = [np.array([2,4]),'hola',3,simulator]
    X = np.array([(1,1, 2), (2,2, 3), (3,3, 4), (4,4, 5), (5,5, 6),(6,6,7),(7,7,8),(8,8,9)])
    print(X.shape)
    Y = np.array([26, 42, 58, 74, 90,106,122,138])
    print(Y.shape)
    print(X@[1,5,10])
    mod = MyLikelihood(endog = Y, exog = X, fixed_params = [1,5])
    res = mod.fit(start_params=[2])
    print('Parameters: alpha = %f; beta = %f; gamma = %f '
          % (1, 5, res.params[0]))
    print('Standard errors: ', res.bse)
    print('P-values: ', res.pvalues)
    print('AIC: %f ' % res.aic)