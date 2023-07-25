#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:39:38 2022

@author: jesus
"""

class LargeCity():
    
    # LARGE CITY CONSTRUCTOR
    def __init__(self,  year,
                 identifier, name,
                 longitud, latitud,
                 minmdt, maxmdt, meanmdt, 
                 mindisn10m, maxdisn10m, meandisn10m, 
                 mincarretn, maxcarretn, meancarretn, 
                 mindisaut, maxdisaut, meandisaut,
                 mindisferr, maxdisferr, meandisferr, 
                 disthospit, 
                 distfarma,
                 distceduc,
                 distcurgh,
                 distatprim,
                 salario,
                 gasto,
                 social,):
        
        self.year = year
        self.population_id = identifier
        self.population_name = name
        
        # FEATURES
        self.longitud    = longitud,
        self.latitud     = latitud
        self.minmdt      = minmdt
        self.maxmdt      = maxmdt
        self.meanmdt     = meanmdt
        self.mindisn10m  = mindisn10m
        self.maxdisn10m  = maxdisn10m
        self.meandisn10m = meandisn10m
        self.mincarretn  = mincarretn
        self.maxcarretn  = maxcarretn
        self.meancarretn = meancarretn
        self.mindisaut   = mindisaut
        self.maxdisaut   = maxdisaut
        self.meandisaut  = meandisaut
        self.mindisferr  = mindisferr
        self.maxdisferr  = maxdisferr
        self.meandisferr = meandisferr
        self.disthospit  = disthospit 
        self.distfarma   = distfarma
        self.distceduc   = distceduc
        self.distcurgh   = distcurgh
        self.distatprim  = distatprim
        self.salario     = salario
        self.gasto       = gasto
        
        self.social = social
        
        self.inhabitants = []


        self.num_men = 0
        self.num_women = 0
        
        
        self.families = {"fam_one_person" : [],
                         "fam_kids"       : []}
        self.families_hist = {}
        
    
    def Print_features(self):
        print('###################################################')
        print('|        Large City (Out-of-the-universe?)        |')
        print('###################################################')
        print("LARGE CITY: %s. Code: %s." 
              % (self.population_name, self.population_id))
        print('--------------------------------------------------')
        print('|                    FEATURES                    |')
        print('--------------------------------------------------')
        print("Latitude:                                %f" % self.latitud)
        print("Longitude:                               %f" % self.longitud)
        print("Min  height above the sea level (m):     %f" % self.minmdt)
        print("Max  height above the sea level (m):     %f" % self.maxmdt)
        print("Mean height above the sea level (m):     %f" % self.meanmdt)
        print("Min  distance to a 10k pop. centre (m):  %f" % self.mindisn10m)
        print("Max  distance to a 10k pop. centre (m):  %f" % self.maxdisn10m)
        print("Mean distance to a 10k pop. centre (m):  %f" % self.meandisn10m)
        print("Min  distance to road (m):               %f" % self.mincarretn)
        print("Max  distance to road (m):               %f" % self.maxcarretn)
        print("Mean distance to road (m):               %f" % self.meancarretn)
        print("Min  distance to highway (m):            %f" % self.mindisaut)
        print("Max  distance to highway (m):            %f" % self.maxdisaut)
        print("Mean distance to highway (m):            %f" % self.meandisaut)
        print("Min  distance to railroad (m):           %f" % self.mindisferr)
        print("Max  distance to railroad (m):           %f" % self.maxdisferr)
        print("Mean distance to railroad (m):           %f" % self.meandisferr)
        print("Mean distance to hospital (m):           %f" % self.disthospit)
        print("Mean distance to pharmacy (m):           %f" % self.distfarma)
        print("Mean distance to education centre (m):   %f" % self.distceduc)
        print("Mean distance to emergency centre (m):   %f" % self.distcurgh)
        print("Mean distance to healthcare centre (m):  %f" % self.distatprim)
        print("Mean annual income:                      %f" % self.salario)
        print("Mean annaual spenditure:                 %f" % self.gasto)
        print("\n")
    