#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:44:44 2022

@author: jesus
"""

import numpy as np
import numpy.random

from PopulationCentre import PopulationCentre

from Family_version_3 import Family
from Family_version_3 import Fam_one_person
from Family_version_3 import Fam_kids

class Agents():
    
    def __init__(self,
                 identifier,
                 sex,
                 age,
                 population_centre,
                 population_others,
                 mdt,
                 carretn,
                 aut,
                 ferr,
                 dis10m,
                 hospi,
                 farma,
                 ceduc,
                 curgh,
                 atprim,
                 salario,
                 gasto,
                 betas,
                 gamma,
                 theta,
                 alphas):
        
        
        # AGENTS/PEOPLE CONSTRUCTOR
        
        self.person_id         = identifier
        self.sex               = sex
        self.age               = age
        self.population_centre = population_centre
        self.population_others = population_others
        
        # RELATED TO FEATURES OF THEIR POPULATION CENTRE
        # Data about the agent's location
        # Height abput the sea level
        self.mdt      = mdt
        # Distance to road
        self.carretn  = carretn
        # Distance to highway
        self.aut      = aut
        # Distance to railroads
        self.ferr     = ferr
        # Distamce to 10k population centre
        self.dis10m   = dis10m
        # Distance to hospital
        self.hospi    = hospi
        # Distamce to pharmacy
        self.farma    = farma
        # Distance to education centre
        self.ceduc    = ceduc
        # Distance to emergency centre
        self.curgh    = curgh
        # Distance to primary healthcare centre
        self.atprim   = atprim
        # Income
        self.salario  = salario
        # Life cost
        self.gasto    = gasto
        
        
        # Following Vietnam THESIS
        # THEORY OF PLANNED BEHAVIOUR -> BEHAVIOURAL ATTITUDE
        self.betas = betas
        self.ba_hist = {}

        # THEORY OF PLANNED BEHAVIOUR -> PERCEIVED BEHAVIOURAL CONTROL
        self.theta = theta
        self.sn_hist = {}
        
        # THEORY OF PLANNED BEHAVIOUR -> PERCEIVED BEHAVIOURAL CONTROL
        self.gamma = gamma
        self.pbc_hist = {}

        # THEORY OF PLANNED BEHAVIOUR -> INTENTIONS
        self.alphas = alphas
        self.intention_hist = {}
        
        # If migrated before
        self.mig_prev = 0
        
        
        
        # Features about the place each person is living in
        self.features = 1
        # Welfare coefficient (I know this coefficient has no much sense
        # but I was trying to create a criterion to decide is s person wants to 
        # migrate or not)
        #if self.features: # non-empty dict
        #    self.happiness = float(np.inner(np.random.dirichlet(np.ones(len(self.features)), size = 1),
        #                     list(self.features.values())) / np.sum(list(self.features.values())))
        #else: # empty dict -> default = 1
        self.happiness = 1
        
        
        ##################### TRYING TO BUILD UP FAMILES #####################
        # Boolean that inidicetes whether an agent is member of a family
        self.family = False
        self.is_kid = None
        self.maybe_parent = None
        ######################################################################
    
    
    def behavioural_attitude(self):
        """
        Theory of planned behaviour: behavioural attitude.
        """
              
        ba = [np.random.uniform(0, x) for x in self.betas]
        ba = [x / sum(ba) for x in ba]
        beta_0 = ba[0]
        beta_1 = ba[1]
        beta_2 = ba[2]
        
        year = self.population_centre.year
        if year in self.population_centre.ba_hist.keys():
            pass
        else:
            self.population_centre.ba_hist[year] = {}
        
        
        #print("Currently living in %s" % self.population_centre.population_name)
        temp = self.population_others.copy()
        temp.remove(self.population_centre)
        #for elem in temp:
        #    print("Other %s" % elem.population_name)
        #print("\n")
        #return None
        factor_0 = beta_0 * self.mdt
    
        factor_1 = beta_1 * np.mean([self.carretn,
                                     self.aut,
                                     self.ferr,
                                     self.dis10m])
    
        factor_2 = beta_2 * np.mean([self.hospi,
                                     self.farma,
                                     self.ceduc,
                                     self.curgh,
                                     self.atprim])
        
        ba_current = factor_0 + factor_1 + factor_2
        
        self.ba_hist[self.population_centre.population_id] = float(ba_current)
        
        if not self.population_centre.population_id in self.population_centre.ba_hist[year].keys():
            self.population_centre.ba_hist[year][self.population_centre.population_id] = [float(ba_current)]
        else:
            self.population_centre.ba_hist[year][self.population_centre.population_id].append(float(ba_current))
        
        
        for elem in temp:
            
            factor_0 = beta_0 * elem.meanmdt
            
            factor_1 = beta_1 * np.mean([np.random.triangular(right = elem.mincarretn, mode =  elem.meancarretn, left = elem.maxcarretn), 
                                         np.random.triangular(right = elem.mindisaut,  mode = elem.meandisaut,   left = elem.maxdisaut),
                                         np.random.triangular(right = elem.mindisferr, mode = elem.meandisferr,  left = elem.maxdisferr),
                                         np.random.triangular(right = elem.mindisn10m, mode = elem.meandisn10m,  left = elem.maxdisn10m),
                                         ])
    
            factor_2 = beta_2 * np.mean([elem.disthospit,
                                         elem.distfarma, 
                                         elem.distceduc,
                                         elem.distcurgh,
                                         elem.distatprim])
    
            ba_current = factor_0 + factor_1 + factor_2
            
            self.ba_hist[elem.population_id] = float(ba_current)
            
            if not elem.population_id in self.population_centre.ba_hist[year].keys():
                self.population_centre.ba_hist[year][elem.population_id] = [float(ba_current)]
            else:
                self.population_centre.ba_hist[year][elem.population_id].append(float(ba_current))
            
        #values = self.ba_hist.values()
        #min_ = min(values)
        #max_ = max(values)
        #self.ba_hist = {key: ((v - min_ ) / (max_ - min_) )  for (key, v) in self.ba_hist.items() } 
            
     
    def subjective_norm(self):
        """
        Theory of planned behaviour: subjective norm.
        
        """
        
        theta_0 = np.random.uniform(0, self.theta)
        
        year = self.population_centre.year
        if year in self.population_centre.sn_hist.keys():
            pass
        else:
            self.population_centre.sn_hist[year] = {}
            
        #my_in  = "IN_" + str(self.age) + "_" + str(self.sex)
        
        my_out_column = "OUT_" + str(self.age) + "_" + str(self.sex)
        my_in_column  = "IN_" + str(self.age) + "_" + str(self.sex)
        
        if my_out_column in list(self.population_centre.social.columns):
            my_out = float(self.population_centre.social[my_out_column])
        else:
            my_out = 0                        
                       
        for destination in self.population_others:
            if my_in_column in list(destination.social.columns):
                my_in =  float(destination.social[my_in_column])
            else:
                my_in = 0
             
            my_res =  theta_0 * (my_out - my_in) * 100
            self.sn_hist[destination.population_id]   = my_res
                
            if not destination.population_id in self.population_centre.sn_hist[year].keys():
                self.population_centre.sn_hist[year][destination.population_id] = [float(my_res)]
            else:
                self.population_centre.sn_hist[year][destination.population_id].append(float(my_res))
            
        #values = self.sn_hist.values()
        #min_ = min(values)
        #max_ = max(values)
        #self.sn_hist = {key: ((v - min_ ) / (max_ - min_) )  for (key, v) in self.sn_hist.items() } 
        
        
        
    
    def perceived_beahavioural_control(self):
        """
        Theory of planned behaviour. perceived behavioural control.
        """
        gamma_0 = np.random.uniform(0, self.gamma)


        year = self.population_centre.year
        if year in self.population_centre.pbc_hist.keys():
            pass
        else:
            self.population_centre.pbc_hist[year] = {}
        
        
        #print("Currently living in %s" % self.population_centre.population_name)
        temp = self.population_others.copy()
        temp.remove(self.population_centre)
        #for elem in temp:
        #    print("Other %s" % elem.population_name)
        #print("\n")
        #return None
        
        #print(float(np.asarray(self.salario)))
        #print(float(np.asarray(self.gasto)))
        #print(float(self.population_centre.distances[str(self.population_centre.population_id)]))
        
        pbc_current = float(np.asarray(self.salario)) - float(np.asarray(self.gasto)) * (1 + (gamma_0 * float(self.population_centre.distances[str(self.population_centre.population_id)])))
        
        self.pbc_hist[self.population_centre.population_id] = float(pbc_current)
        
        if not self.population_centre.population_id in self.population_centre.pbc_hist[year].keys():
            self.population_centre.pbc_hist[year][self.population_centre.population_id] = [float(pbc_current)]
        else:
            self.population_centre.pbc_hist[year][self.population_centre.population_id].append(float(pbc_current))
        
        
        for elem in temp:
            pbc_current = float(np.asarray(self.salario)) - float(np.asarray(self.gasto)) * (1 + (gamma_0 * float(self.population_centre.distances[str(elem.population_id)])))
            
            self.pbc_hist[elem.population_id] = float(pbc_current)
            
            if not elem.population_id in self.population_centre.pbc_hist[year].keys():
                self.population_centre.pbc_hist[year][elem.population_id] = [float(pbc_current)]
            else:
                self.population_centre.pbc_hist[year][elem.population_id].append(float(pbc_current))
                
        #print("\n")
        #values = self.pbc_hist.values()
        #min_ = min(values)
        #max_ = max(values)
        #self.pbc_hist = {key: ((v - min_ ) / (max_ - min_) )  for (key, v) in self.pbc_hist.items() }
   

      
   
    
    
    def intention(self):
        """
        Theory of planned behaviour: intention
        """
        year = self.population_centre.year
        if year in self.population_centre.intention_hist.keys():
            pass
        else:
            self.population_centre.intention_hist[year] = {}
        
        for key in self.ba_hist.keys():
            temp = (self.alphas[0] * self.ba_hist[key]) + (self.alphas[1] * self.sn_hist[key]) + (self.alphas[2] * self.pbc_hist[key])
            self.intention_hist[key] = temp
            
            if not key in self.population_centre.intention_hist[year].keys():
                self.population_centre.intention_hist[year][key] = [float(temp)]
            else:
                self.population_centre.intention_hist[year][key].append(float(temp))
        
        self.intention_hist = dict(sorted(self.intention_hist.items(), key=lambda item: item[1]))
        
        
        
        #values = self.intention_hist.values()
        #min_ = min(values)
        #max_ = max(values)
        #self.intention_hist = {key: ((v - min_ ) / (max_ - min_) )  for (key, v) in self.intention_hist.items() }

            
        
    """
    def migrate(self):
        # If a person want to migrate
        # If the peson is "unhappy" (?)
        if self.happiness <= 0.15:
            # If the person is old enough (just trying to model what we said about families..)
            # I think we will are able to "create" famillies but i have 
            # doubts about some of variables (i will try later)
            if self.age >= 18: # better 18
                self.new_migration = 1
                self.migration = self.new_migration
                # That person is leaving the population centre 
                # but is the person leaving the universe ???????????
                #self.remove_agent()
                b = True
            else:
                b = False
        else:
            b = False
        return b
    """
    
    
    def update_infra(self):
        # Height abpve the sea level
        self.mdt      = np.random.triangular(right = self.population_centre.minmdt,
                                             mode  = self.population_centre.meanmdt, 
                                             left  = self.population_centre.maxmdt)
        # Distance to road
        self.carretn  = np.random.triangular(right = self.population_centre.mincarretn,
                                             mode  = self.population_centre.meancarretn, 
                                             left  = self.population_centre.maxcarretn)
        # Distance to highway
        self.aut      = np.random.triangular(right = self.population_centre.mindisaut,
                                             mode  = self.population_centre.meandisaut, 
                                             left  = self.population_centre.maxdisaut)
        # Distance to railroads
        self.ferr     = np.random.triangular(right = self.population_centre.mindisferr,
                                             mode  = self.population_centre.meandisferr,
                                             left  = self.population_centre.maxdisferr)
        # Distamce to 10k population centre
        self.dis10m   = np.random.triangular(right = self.population_centre.mindisn10m,
                                             mode  = self.population_centre.meandisn10m,
                                             left  = self.population_centre.maxdisn10m)
        # Distance to hospital
        self.hospi    = self.population_centre.disthospit
        # Distamce to pharmacy
        self.farma    = self.population_centre.distfarma
        # Distance to education centre
        self.ceduc    = self.population_centre.distceduc
        # Distance to emergency centre
        self.curgh    = self.population_centre.distcurgh
        # Distance to primary healthcare centre
        self.atprim   = self.population_centre.distatprim
    
    
    
    def update_eco(self, df_eco_mun, df_eco_atr):
        
        #print("LOOKINF FOR %s" % self.population_centre.population_id)
        
        if self.population_centre.population_id in list(df_eco_mun["CODMUN"]):
            #print("ENCONTRADO EN MUN")
            df_temp_3 = df_eco_mun.query('CODMUN == ' + str(self.population_centre.population_id))
        
        elif self.population_centre.population_id in list(df_eco_atr["CODMUN"]):
            #print("ENCONTRADO EN ATR")
            df_temp_3 = df_eco_atr.query('CODMUN == ' + str(self.population_centre.population_id))
        
        
        #print("\n")
        self.salario     = df_temp_3["SALARIO_MEAN_" + str(self.population_centre.year)].values,
        # Cost of living
        self.gasto       = df_temp_3["GASTO_MEAN_" + str(self.population_centre.year)].values,
        
        

    ####################### TRYING TO BUILD UP FAMILES #######################
    # When updating, roles change
    def family_role(self):
        if self.age < 18:
            self.is_kid = True
            self.maybe_parent = False
        elif 18 <= self.age <= 60:
            self.is_kid = False
            self.maybe_parent = True
        else:
            self.is_kid = False
            self.maybe_parent = False
    ##########################################################################
        
    
    def add_agent(self, new = True):
        # Add agent to population centre
        self.population_centre.inhabitants.append(self)
        if new:
            if self.sex == "M":
                self.population_centre.num_men += 1
            else:
                self.population_centre.num_women += 1
                
        
            
    def remove_agent(self):
        # Remove agent from population centre
        self.population_centre.inhabitants.remove(self)
        if self.sex == "M":
            self.population_centre.num_men -= 1
        else: # sex = "F"
            self.population_centre.num_women -= 1
    
    def die(self):
        self.remove_agent()
            
        

        
    def Print(self):
        print('- - - - - - - - - - - - - - - - - - - - - - - - - -')
        print('|                      AGENT                      |')
        print('- - - - - - - - - - - - - - - - - - - - - - - - - -')
        print("Lives in %s" % self.population_centre.population_name)
        print("Agent id: %s" % self.person_id)
        print("Age: %s" % self.age)
        print("Sex: %s" % self.sex)
        print("Happiness: %s" % self.happiness)
        print("\n")
        
        
        
        