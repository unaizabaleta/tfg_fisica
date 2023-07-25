#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:26:12 2022

@author: jesus
"""

from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from Universe import Universe

from joblib import dump, load

 

import tkinter as tk
from tkinter import messagebox
import sys
from PIL import ImageTk, Image
from paho.mqtt.client import Client 
from time import sleep

from PopulationCentre import PopulationCentre
from LargeCity import LargeCity
from Agents import Agents
from Family_version_3 import Fam_one_person
from Family_version_3 import Fam_kids


from SeaofBTCapp import Pages
from SeaofBTCapp import SeaofBTCapp
from SeaofBTCapp import StartPage
from SeaofBTCapp import PageOne
from SeaofBTCapp import PopulationCentrePage
from SeaofBTCapp import YearsPage
from SeaofBTCapp import PlotPage

import pandas as pd
import numpy as np
  
          
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from  matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import plotly.offline as py
import sys



class MyLikelihood(GenericLikelihoodModel):
    
    def __init__(self, endog, todo, **kwds):
        self.todo = todo
        super(MyLikelihood, self).__init__(endog, endog, **kwds)

    def loglike(self, params):
        print(params)
        Y = self.endog
        X = self.todo
        my_universe = Universe(year                     = X[0],
                           df_historic_ages             = X[1],
                           df_families                  = X[2],
                           df_features                  = X[3],
                           df_income_spend              = X[4],
                           df_features_large_cities     = X[5],
                           df_income_spend_large_cities = X[6],
                           df_social                    = X[7],
                           df_distances                 = X[8],
                           betas  = X[9],
                           gamma  = X[10],
                           theta  = X[11],
                           alphas = params,
                           natality_model  = X[13],
                           mortality_model = X[14])
        aux = my_universe.get_poblacion_por_mun()
        for i in range(11):
            my_universe.update()
            aux = np.append(aux,my_universe.get_poblacion_por_mun())
        Y_pred = aux
        # Error cuadrático
        #res = np.sum((Y_pred - Y)**2)
        # Lo ponderamos por tamaño de la poblacion
        res = np.mean((Y_pred-Y)**2 / np.sqrt(Y))
        print(res)
        return -res
        
    
     
if __name__ == "__main__":
    
    year             = 2010
    
    # COMARCA 2
    path             = "Dominio/Comarca_2/Normal_wrt_Comarca_2/"
    df_historic_ages = pd.read_csv(path + "df_2_historic_ages.csv")
    df_families      = pd.read_csv(path + "df_2_families.csv")
    df_features      = pd.read_csv(path + "df_2_infra_coords_normal.csv")
    df_income_spend  = pd.read_csv(path + "df_2_income_spend_normal.csv") 
    df_distances     = pd.read_csv(path + "df_2_distances.csv", index_col = 0).fillna(0)
    
    # Large Cities
    #path             = "Dominio/Comarca_2/"
    #df_features_large_cities     = pd.read_csv(path + "df_large_cities_infra_coords_normal.csv")
    #df_income_spend_large_cities = pd.read_csv(path + "df_large_cities_income_spend_normal.csv")
    
    # Subjective norm (social)
    path             = "Dominio/Social_Norm/"
    df_social = pd.read_csv(path + "df_social.csv")
    
    # Natality and mortality models
    path = "Modelos/"
    #keras.models.load_model
    natality_model  = load(path + "natality_model_subset_linreg.joblib")
    mortality_model = load(path + "mortality_model_subset_linreg.joblib")
    
    # betas: list of 3
    # beta[0] -> slope and height above the sea
    # beta[1] -> distance 10k, road, highway, railroad, 
    # beta[2] -> hostital, pharmacy, education, emergency, healthcare 
    
    # gamma: parameter for subjective norm
    
    
    Y = [1]
    X = [1]
    mod = MyLikelihood(endog = Y, exog = X)
    res = mod.fit(
            start_params = np.array(np.random.uniform(0, 1, 8)),
            maxiter = 1)
