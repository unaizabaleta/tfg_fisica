#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 21:52:38 2022

@author: jesus
"""

from joblib import dump, load
import keras
import tensorflow as tf
 

import tkinter as tk
from tkinter import messagebox
import sys
from PIL import ImageTk, Image
from paho.mqtt.client import Client 
from time import sleep

from Universe import Universe
from Likelihood_test import MyLikelihood
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
import csv

     
if __name__ == "__main__":
    
    year             = 2010
    
    # COMARCA 2
    path             = "Dominio/Comarca_2_atractor/Comarca_2_aislado/"
    df_historic_ages = pd.read_csv(path + "df_2_atractor_aislado_historic_ages.csv")
    df_families      = pd.read_csv(path + "df_2_atractor_aislado_families.csv")
    df_features      = pd.read_csv(path + "df_2_atractor_aislado_infra_coords_normal.csv")
    df_income_spend  = pd.read_csv(path + "df_2_atractor_aislado_income_spend_normal.csv") 
    
    # DISTANCES
    path             = "Dominio/Comarca_2_atractor/"
    df_distances     = pd.read_csv(path + "df_2_atractor_distances.csv", index_col = 0).fillna(0)
    df_pueblos = pd.read_csv(path + "df_2_atractor_historic_ages.csv", index_col = 0)
    df_pueblos = df_pueblos[df_pueblos['Nombre']!='Santander']
    print(df_pueblos.shape)
    # Large Cities
    path             = "Dominio/Comarca_2_atractor/Atractor_aislado/"
    df_features_large_cities     = pd.read_csv(path + "df_2_atractor_aislado_infra_coords_normal.csv")
    df_income_spend_large_cities = pd.read_csv(path + "df_2_atractor_aislado_income_spend_normal.csv")
    
    # Subjective norm (social)
    path             = "Dominio/Subjective_Norm/"
    df_social = pd.read_csv(path + "df_subjective_norm_temp.csv").fillna(0)
    
    # Natality and mortality models
    path = "Modelos/"
    # keras.models.load_model
    natality_model = keras.models.load_model(path + "natality_model_subset_ann.h5")
    mortality_model = load(path + "mortality_model_subset_linreg_clases.joblib")
    
    # betas: list of 3
    # beta[0] -> slope and height above the sea
    # beta[1] -> distance 10k, road, highway, railroad, 
    # beta[2] -> hostital, pharmacy, education, emergency, healthcare 
    
    # gamma: parameter for subjective norm
    
    
    #betas = list(np.random.uniform(0, 1, 3))
    betas = [0.40848788491878807, 0.02758170121289083, 0.4398889209246927]
    #gamma = np.random.uniform(0, 1)
    gamma = 0.65
    #theta = np.random.uniform(0, 1)
    theta = 0.65 #sn
    #alphas = list(np.random.uniform(0, 1, 3))
    alphas = [ 0.002211, 0.004979, 0.004753]
    
    print("BETAS %s" % betas)
    print("GAMMA %s" % gamma)
    print("THETA %s" % theta)
    print("ALPHAS %s" % alphas)
    
    my_txt = open("pruebas/prueba21/params.txt", "w")
    
    for elem in betas:
        my_txt.write(str(elem) + "\n")
        
    my_txt.write(str(gamma)+ "\n")
    my_txt.write(str(theta)+ "\n")
    
    for elem in alphas:
        my_txt.write(str(elem) + "\n")
        
    my_txt.close()
        

    
    with open("pruebas/prueba21/total.csv", "a", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(["TransactionAmt", "Source", "Target", "Date"])
                
    with open("pruebas/prueba21/kids.csv", "a", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(["TransactionAmt", "Source", "Target", "Date"])
                
    with open("pruebas/prueba21/unip.csv", "a", newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(["TransactionAmt", "Source", "Target", "Date"])

    # -----------------------------------------------------------------------------------
    # Aquí metemos toda la calibración
    print("---------- Calibración ----------------")
    X = [year,df_historic_ages,df_families,df_features,df_income_spend,df_features_large_cities,df_income_spend_large_cities,df_social,df_distances,betas,gamma,theta,alphas,natality_model,mortality_model]
    #Ahora hay que generar el Y_test


    Y_df = df_pueblos[['POB2010','POB2011','POB2012','POB2013','POB2014','POB2015','POB2016','POB2017','POB2018','POB2019','POB2020','POB2021']].values
    Y = np.reshape(np.transpose(Y_df),-1)
    mod = MyLikelihood(endog = Y, todo = X)
    res = mod.fit(method='lbfgs',start_params=[0.30769482, 0.00347893, 0.00347893],m=3,bounds=[(0,1),(0,1),(0,1)])


    print('Parameters: gamma %f'
         % res.params )
    print('Standard errors: ', res.bse)
    print('P-values: ', res.pvalues)
    print('AIC: %f ' % res.aic)

    # Meter el input (que serán biende cosas, igual se pueden meter separadas y luego
    # -----------------------------------------------------------------------------------
    my_universe = Universe(year                         = year,
                           df_historic_ages             = df_historic_ages,
                           df_families                  = df_families,
                           df_features                  = df_features,
                           df_income_spend              = df_income_spend,
                           df_features_large_cities     = df_features_large_cities,
                           df_income_spend_large_cities = df_income_spend_large_cities,
                           df_social                    = df_social,
                           df_distances                 = df_distances,
                           betas  = betas,
                           gamma  = gamma, 
                           theta  = 0.661,
                           alphas = alphas,
                           natality_model  = natality_model,
                           mortality_model = mortality_model)
    aux = my_universe.get_poblacion_por_mun()
    for i in range(11):
        my_universe.update()
        aux = np.append(aux, my_universe.get_poblacion_por_mun())
    Y_pred = aux
    #print(Y)
    print(Y_pred)
    #print(np.sum((Y-Y_pred)**2))
    
    app = SeaofBTCapp(universe = my_universe)
    app.mainloop()
       
