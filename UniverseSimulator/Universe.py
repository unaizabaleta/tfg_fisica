#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:43:03 2022

@author: jesus
"""

# Load model architecture
from PopulationCentre import PopulationCentre
from LargeCity import LargeCity
from Agents import Agents
from Family_version_3 import Family
from Family_version_3 import Fam_one_person
from Family_version_3 import Fam_kids

# Load regression metrics
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score                         

# Queues for families
from collections import deque

# Load basic libraries
import pandas as pd
import random
import numpy as np
import random, time, math, sys
import warnings
import re
from varname import nameof
from itertools import chain
import csv
from statistics import mean

# Load libraries for plots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as py

import io
import contextlib

# Libraries for distances
import geopy.distance

# Load libraries for warnings
import warnings
warnings.simplefilter("always")



def myround(x, base=5):     
    """
    Auxiliary function. Given an age, returns its range according to
    the discretization in the read data.
        
    Examples
    ------
    >>> myround(1)
    0-4        
    >>> myround(23)
    20-24        
    >>> myround(106)
    >100
    """
    init = base * math.floor(float(x) / base)    
    if init >= 100:
        return '>' + str(100)     
    end =  base * math.ceil(float(x) / base)
    if init == end:
        end = end + 5
    return str(init) + '-' + str(end - 1)

        

class Universe():
    # MAIN CLASS
    
    def __init__(self,
                 year,
                 df_historic_ages,
                 df_families,
                 df_features,
                 df_income_spend,
                 df_features_large_cities,
                 df_income_spend_large_cities,
                 df_social,
                 df_distances,
                 # TPB: Behavioural attitude
                 betas, #list/array of 11
                 # TPB: Perceived behavioural control
                 gamma, #float
                 # TPB: Subjective norm
                 theta, #float
                 # TPB: Intention
                 alphas, #list/array of 3
                 natality_model,
                 mortality_model
                 ):


        # CONSTRUCTOR
        global agent_idx
        agent_idx = 0
        # Year
        self.year = str(year)
        #################### NATALITY AND MORTALITY MODELS ####################
        self.natality_model  = natality_model
        self.mortality_model = mortality_model
        
        
        # Read data from dataframe (population,......)
        self.df_historic_ages = df_historic_ages
        
        # Theory of planned behaviour
        # Behavioural attitue
        self.betas = betas
        self.gamma = gamma
        self.theta = theta
        self.alphas = alphas
        
        ##################### TRYING TO BUILD UP FAMILES #####################
        # Read data from dataframe (FAMILIES)
        self.df_families     = df_families
        self.df_features     = df_features
        self.df_income_spend = df_income_spend 
        self.df_social       = df_social 
        self.df_distances    = df_distances 
        ######################################################################
        

        # List of population centres (nucleos de poblacion) if the universe
        population_centres = self.PopulationCentreBuilder()
        self.population_centres  = population_centres[0]
        self.cols_update = population_centres[1]
        
        # LARGE CITIES

        self.df_features_large_cities = df_features_large_cities
        self.df_income_spend_large_cities = df_income_spend_large_cities
        self.large_cities = self.LargeCityBuilder()
        
        
        # List of persons in the universe
        self.universe_persons = self.AgentsBuilder

        ##################### TRYING TO BUILD UP FAMILES #####################
        self.FamilyBuilder()
        ######################################################################
        
        

        
        

    def PopulationCentreBuilder(self):
        # METHOD TO BUILD UP POPULATION CENTRES
        # List to store population centres
        population_centres = [] 
        # As we are reading from a datafrase (assume each population centre 
        # appears just once in the dataframe), consider each row:
        for population in range(self.df_historic_ages.shape[0]):
            
            
            # Select specific row, i..e. specific population centre
            df_temp      = self.df_historic_ages.iloc[population]
            identifier   = df_temp["CODMUN"]
            
            df_temp_2    = self.df_features.\
                                query('CODMUN == ' + str(identifier))
            
            df_temp_3    = self.df_income_spend.\
                                query('CODMUN == ' + str(identifier))
            
            df_distances = self.df_distances.loc[[identifier]]
            
            if identifier in list(self.df_social["CODE"]):
                df_social = self.df_social.loc[self.df_social["CODE"] == identifier]
            else:
                df_social = self.df_social.loc[self.df_social["CODE"] == 39]
            
            #my_cols   = ["HOM" + self.year, "MUJ" + self.year,
            #             "NAT" + self.year, "MOR" + self.year, 
            #            "SALDOTT" + self.year]
            my_cols   = ["NAT" + self.year, "MOR" + self.year, "POB" + self.year]
            
            #my_cols_update = ["NAT", "MOR", "SALDOTT"]
            #my_cols_update = ["NAT", "MOR"]
            
            d_args = {}
            for column in my_cols:
                d_args[column[:len(column)-4].lower()] = df_temp[column]
            
           
            
            # Invoke Population Center constructor
            the_population = PopulationCentre(
                    year        = self.year,
                    identifier  = identifier,
                    name        = df_temp["Nombre"],
                    num_men     = 0,
                    num_women   = 0,
                    # Select features for the population centre
                    # To compute geodesic distance ...
                    longitud    = df_temp_2["LONGITUD_E"],
                    latitud     = df_temp_2["LATITUD_ET"],
                    # Height about the sea level
                    minmdt      = df_temp_2["MINMDT"],
                    maxmdt      = df_temp_2["MAXMDT"] ,
                    meanmdt     = df_temp_2["MEANMDT"],
                    # Distance to 10k population centre
                    mindisn10m  = df_temp_2["MINDISN10M"],
                    maxdisn10m  = df_temp_2["MAXDISN10M"],
                    meandisn10m = df_temp_2["MEANDISN10M"],
                    # Distance to road
                    mincarretn  = df_temp_2["MINCARRETN"],
                    maxcarretn  = df_temp_2["MAXCARRETN"],
                    meancarretn = df_temp_2["MEANCARRETN"],
                    # Distance to highway
                    mindisaut   = df_temp_2["MINDISAUT"],
                    maxdisaut   = df_temp_2["MAXDISAUT"],
                    meandisaut  = df_temp_2["MEANDISAUT"],
                    # Distance to railroad
                    mindisferr  = df_temp_2["MINDISFERR"],
                    maxdisferr  = df_temp_2["MAXDISFERR"],
                    meandisferr = df_temp_2["MEANDISFERR"],
                    # Distance to hospitals
                    disthospit  = df_temp_2["DISTHOSPIT"],
                    # Distance to pharmacies
                    distfarma   = df_temp_2["DISTFARMA"],
                    # Distance to education centres
                    distceduc   = df_temp_2["DISTCEDUC"],
                    # Distance to emercengy centres
                    distcurgh   = df_temp_2["DISTCURGH"],
                    # Distance to primary healthcare centres
                    distatprim  = df_temp_2["DISTATPRIM"],
                    # Subjective / social norm
                    social      = df_social, 
                    natality    = 0,
                    mortality   = 0,
                    # Data about income and spenditures
                    salario     = df_temp_3["SALARIO_MEAN_" + str(self.year)], 
                    gasto       = df_temp_3["GASTO_MEAN_"   + str(self.year)],
                    distances   = df_distances,
                    **d_args)
            
    
           
            # Add specific population to the universe
            population_centres.append(the_population)

        return  [population_centres, ["NAT", "MOR", "POB"]]
    
    
    def LargeCityBuilder(self): 
        # METHOD TO BUILD UP LARGE CITIES
        # List to store population centres
        large_cities = [] 
        # As we are reading from a datafrase (assume each population centre 
        # appears just once in the dataframe), consider each row:
        for city in range(self.df_features_large_cities.shape[0]):
            
            # Select specific row, i..e. specific population centre
            df_temp_2  = self.df_features_large_cities.iloc[city]
            identifier = df_temp_2["CODMUN"]
            df_temp_3  = self.df_income_spend_large_cities.\
                                query('CODMUN == ' + str(identifier))
            
            
            if identifier in list(self.df_social["CODE"]):
                df_social = self.df_social.loc[self.df_social["CODE"] == identifier]
            else:
                df_social = self.df_social.loc[self.df_social["CODE"] == 39]
            
        
            # Invoke Population Center constructor
            the_population = LargeCity(
                    year        = self.year,
                    identifier  = df_temp_2["CODMUN"],
                    name        = df_temp_2["NOMBRE"],
                    # Select features for the population centre
                    # To compute geodesic distance ...
                    longitud    = df_temp_2["LONGITUD_E"],
                    latitud     = df_temp_2["LATITUD_ET"],
                    # Height about the sea level
                    minmdt      = df_temp_2["MINMDT"],
                    maxmdt      = df_temp_2["MAXMDT"] ,
                    meanmdt     = df_temp_2["MEANMDT"],
                    # Distance to 10k population centre
                    mindisn10m  = df_temp_2["MINDISN10M"],
                    maxdisn10m  = df_temp_2["MAXDISN10M"],
                    meandisn10m = df_temp_2["MEANDISN10M"],
                    # Distance to road
                    mincarretn  = df_temp_2["MINCARRETN"],
                    maxcarretn  = df_temp_2["MAXCARRETN"],
                    meancarretn = df_temp_2["MEANCARRETN"],
                    # Distance to highway,
                    mindisaut   = df_temp_2["MINDISAUT"],
                    maxdisaut   = df_temp_2["MAXDISAUT"],
                    meandisaut  = df_temp_2["MEANDISAUT"],
                    # Distance to railroad
                    mindisferr  = df_temp_2["MINDISFERR"] ,
                    maxdisferr  = df_temp_2["MAXDISFERR"] ,
                    meandisferr = df_temp_2["MEANDISFERR"],
                    # Distance to hospitals
                    disthospit  = df_temp_2["DISTHOSPIT"],
                    # Distance to pharmacies
                    distfarma   = df_temp_2["DISTFARMA"],
                    # Distance to education centres
                    distceduc   = df_temp_2["DISTCEDUC"],
                    # Distance to emercengy centres
                    distcurgh   = df_temp_2["DISTCURGH"],
                    # Distance to primary healthcare centres
                    distatprim  = df_temp_2["DISTATPRIM"],
                    # Income
                    salario     = df_temp_3["SALARIO_MEAN_" + str(self.year)],
                    # Cost of living
                    gasto       = df_temp_3["GASTO_MEAN_" + str(self.year)],
                    social      = df_social,)
                    

            # Add specific population to the universe
            large_cities.append(the_population)
        
        return large_cities
        
        

    
    
    @property
    def AgentsBuilder(self):
        # Method to build up agents
        global agent_idx
        agents = []
        age_cols = [col for col in self.df_historic_ages.columns if col.startswith('Edad_')]
        #age_cols =self.df_historic_ages.filter(regex = r'^Edad_.*?$', axis = 1)
        #male_cols =  self.df_historic_ages.filter(regex = r'^Edad_M.*?$', axis = 1)
        #female_ cols = self.df_historic_ages.filter(regex = r'^Edad_F.*?$', axis = 1)
        for population in self.population_centres:
            # Dictionary for age ranges
            age_range = {self.year + "M" : {}, self.year + "F" : {}}
            # Select subdataframe
            df_temp = self.df_historic_ages.\
                query('CODMUN == ' + str(population.population_id))[age_cols]
                
            for col in df_temp.columns:
                sex = col.split(":")[0][-1]
                if "-" in col:
                    init = int(col.split(":")[1].split("-")[0])
                    end  = int(col.split(":")[1].split("-")[1])
                    key  = str(init) + '-' + str(end)
                else:
                    init = int(col.split(":")[1].split(">")[1])
                    end  = 110
                    key  = ">" + str(init)
                    
                # Update dictionay for age ranges
                if key not in age_range[self.year + sex].keys():
                    age_range[self.year + sex].update({key : int(df_temp[col])})
                else: 
                    age_range[self.year + sex][key] += int(df_temp[col])
                        
                for i in range(int(df_temp[col])):
                               
                    # Crate agent
                    # Other values? distributions?

                    the_agent = Agents(
                            identifier        = agent_idx,
                            sex               = sex,
                            age               = random.randint(init, end),
                            population_centre = population,
                            population_others = [*self.population_centres, *list(self.large_cities)],
                            mdt               = np.random.triangular(right = population.minmdt,     mode = population.meanmdt,      left = population.maxmdt),
                            carretn           = np.random.triangular(right = population.mincarretn, mode =  population.meancarretn, left = population.maxcarretn),
                            aut               = np.random.triangular(right = population.mindisaut,  mode = population.meandisaut,   left = population.maxdisaut),
                            ferr              = np.random.triangular(right = population.mindisferr, mode = population.meandisferr,  left = population.maxdisferr),
                            dis10m            = np.random.triangular(right = population.mindisn10m, mode = population.meandisn10m,  left = population.maxdisn10m),
                            hospi             = population.disthospit,
                            farma             = population.distfarma,
                            ceduc             = population.distceduc,
                            curgh             = population.distcurgh,
                            atprim            = population.distatprim,
                            salario           = population.salario,
                            gasto             = population.gasto,
                            betas             = self.betas,
                            gamma             = self.gamma,
                            alphas            = self.alphas,
                            theta             = self.theta,
                            )
                    
                    ############### TRYING TO BUILD UP FAMILES ###############
                    the_agent.family_role()
                    ##########################################################

                    # Add agent to population centre
                    the_agent.add_agent()
                    # Add agent to global list
                    agents.append(the_agent)
                    # Update identifier
                    agent_idx += 1
            
            # Update dictionary with age ranges and historial
            population.ages_hist = age_range
            
            df = pd.DataFrame.from_dict(population.ages_hist)        
            my_cols = [col for col in df.columns if str(self.year) in col]
            df = df[my_cols]
            temp = df.transpose()
            temp_male = pd.DataFrame(temp.iloc[0]).transpose()
            temp_male.columns =["H" + x.replace("-", "A") if "-" in x else "H100MS" for x in temp_male.columns]
            temp_female = pd.DataFrame(temp.iloc[1]).transpose()
            temp_female.columns = ["M" + x.replace("-", "A") if "-" in x else "M100YMS" for x in temp_female.columns]
            df_X = pd.concat([temp_male.reset_index(drop=True),
                          temp_female.reset_index(drop=True)], axis=1)
    
            df_X_natality = df_X[['H0A4', 'H5A9', 'H10A14', 'H15A19', 'H20A24', 'H25A29', 'H30A34',
       'H35A39', 'H40A44', 'H45A49', 'H50A54', 'H55A59', 'M0A4', 'M5A9',
       'M10A14', 'M15A19', 'M20A24', 'M25A29', 'M30A34', 'M35A39', 'M40A44',
       'M45A49', 'M50A54', 'M55A59']]
            
            df_X_mortality = df_X[['H60A64', 'H65A69', 'H70A74', 'H75A79', 'H80A84', 'H85A89', 'H90A94',
       'H95A99', 'H100MS', 'M60A64', 'M65A69', 'M70A74', 'M75A79', 'M80A84',
       'M85A89', 'M90A94', 'M95A99', 'M100YMS']]
            mortality = self.mortality_model.predict(df_X_mortality)

            natality  = self.natality_model.predict(df_X_natality, verbose = None)
           
            
            mortality = 0 if mortality <= 0 else np.rint(mortality)
            natality  = 0 if natality  <= 0 else np.rint(natality)
            
            population.natality  = natality
            population.mortality = mortality


            # Update historial for ages
            population.update_population(iteration = 0)
            population.update_population_hist()
            # Update historial for families
            population.update_families_hist()
            
            
        return agents
            
    
    # Different approach
    def FamilyBuilder(self):
        
        # Consider each population_centre
        for population in self.population_centres:
            
            # Given a population centre, select specific row in df
            df_temp = self.df_families.\
                query('CODMUN == ' + str(population.population_id))
                
            # Number of kids for each population centre:
            num_kids = 0
            for key in list(population.ages_hist.keys()):
                for key_2 in population.ages_hist[key].keys():
                    if "-" in key_2:
                        if int(key_2.split("-")[1]) < 25:
                            num_kids += population.ages_hist[key][key_2]
                            
            
            ##### 3-4-5 people families #####
            # fam3p -> father + mother + kidx1
            fam3p = df_temp["3PER"].values
            # fam4p -> father + mother + kidx2
            fam4p = df_temp["4PER"].values
            # fam5p -> father + mother + kidx3
            fam5p = df_temp["5PER"].values
            
            # Total families with kids
            fam = fam3p + fam4p + fam5p
            
            #print(population.population_name)
            #print("kids  %f" % num_kids)
            #print("fam3p %f" % fam3p)
            #print("fam4p %f" % fam4p)
            #print("fam5p %f" % fam5p)
            #print("fam   %f" % fam)
            
            # Percentage of each type
            fam3p = math.ceil(num_kids * (fam3p / fam))
            fam4p = math.ceil(num_kids * (fam4p / fam))
            fam5p = math.ceil(num_kids * (fam5p / fam))
            
            # UNCOMMIT TO CHECK RESULTS
            #print("Nº of families with 1 kid : %s"  % fam3p)
            #print("Nº of families with 2 kids: %s" % int(math.ceil(fam4p / 2)))
            #print("Nº of families with 3 kids: %s" % int(math.ceil(fam5p / 3)))
            temp = fam3p + int(math.ceil(fam4p / 2))*2 + int(math.ceil(fam5p / 3))*3
            #print("Space for %s kids" % temp)
            #print("\n")
            
            # Declare queue for families
            queue_families = deque()
            
            # Create families
            for i in range(fam3p):
                # Build up family
                fam = Fam_kids(population_centre = population, kids_limit = 1)    
                # Append left to queue
                queue_families.appendleft(fam)
            
            for i in range(int(math.ceil(fam4p / 2))):
                # Build up family
                fam = Fam_kids(population_centre = population, kids_limit = 2)
                # Append left to queue
                queue_families.appendleft(fam)
            
            for i in range(int(math.ceil(fam5p / 3))):
                # Build up family
                fam = Fam_kids(population_centre = population, kids_limit = 3)
                # Append left to queue
                queue_families.appendleft(fam)
        
        
            # PROBLEMA DE ESTA INICIALIZACION: HIJOS CON EDADES MUY HOMOGENEAS
            # Search for kids
            for agent in population.inhabitants:
            
                # If the agent is neither a kid nor a parent
                if (not agent.is_kid) and (not agent.maybe_parent):
                    # Build up one person family
                    my_family = Fam_one_person(population)
                    my_family.update(agent)
                    population.families["fam_one_person"].append(my_family)
                        
                    # If the agent is a kid
                elif agent.is_kid and (not agent.family):
                    # Consider first family in queue
                    # If there's room for the kid
                    if len(queue_families[0].kids) < queue_families[0].kids_limit:
                        queue_families[0].update(agent, "kid")
                    # If there's no room for the kid
                    else:
                        # The family has as many kids as possible so add to the universe
                        population.families["fam_kids"].append(queue_families[0])
                        # and remode from the queue
                        queue_families.popleft()
                        # The kid must be added to the next family
                        queue_families[0].update(agent, "kid")
            # In case there are any families in the queu, move them to the universe
            while len(queue_families) > 0:
                if len(queue_families[0].kids) > 0:
                    population.families["fam_kids"].append(queue_families[0])
                queue_families.popleft()
        

            # Search for parents
            for agent in population.inhabitants:
                if agent.maybe_parent and (not agent.family):
                    # Consider each family with kids
                    for family in population.families["fam_kids"]:
                        my_bool = True
                        # If the agent is a male
                        if agent.sex == "M":
                            # If the family has no father
                            if not family.father:
                                # Check parents/kids ages are compatible
                                for kid in family.kids:
                                    my_bool = agent.age >= kid.age + 25
                        
                                if my_bool:
                                    # If theres no mother
                                    if not family.mother:
                                        family.update(agent, "father")
                                        break
                                    else: # Verify ages
                                        my_bool = (family.mother.age - 5 <= agent.age <= family.mother.age - 5) or (agent.age - 5 <= family.mother.age <= agent.age + 5)
                                        if my_bool:
                                            family.update(agent, "father")
                                            break
                                        else:
                                            pass
                                    
                    
                        else: #agent.sex = "F"
                            if not family.mother:
                                # If the family has no father
                                for kid in family.kids:
                                    # Check parents/kids ages are compatible
                                    my_bool = agent.age >= kid.age + 25
                        
                                if my_bool:
                                    # if theres no father
                                    if not family.father:
                                        family.update(agent, "mother")
                                        break
                                    else: # verify ages
                                        my_bool = (family.father.age - 5 <= agent.age <= family.father.age + 5) or (agent.age - 5 <= family.father.age <= agent.age + 5)
                                        if my_bool:
                                            family.update(agent, "mother")
                                            break
                                        else:
                                            pass
                                
            
                # agent is neither compatible with kids or partner
                if not agent.family:
                    my_family = Fam_one_person(population)
                    my_family.update(agent)
                    population.families["fam_one_person"].append(my_family)
                                    
            # Update families hist
            population.update_families_hist()
        
        
    ###########################################################################
    
    
            
        
                    
    def update(self):
        global agent_idx
        # Update year for Universe
        self.year = str(int(self.year) + 1)
        # Consider each population centre
        for population in self.population_centres:
            # Intialize dictoinary with age ranges to previous year
            population.ages_hist[self.year + "M"] = population.ages_hist[str(int(self.year) - 1) + "M"].copy()
            population.ages_hist[self.year + "F"] = population.ages_hist[str(int(self.year) - 1) + "F"].copy()
            
            
        for population in self.population_centres:
            #print("INICIO ACTUALIZACION")
            #print("HOMBRES")
            #print(population.ages_hist[self.year + "M"])
            #print("\n")
            #print("MUJERES")
            #print(population.ages_hist[self.year + "F"])
            #print("\n")

            
            ### PEOPLE WHO LEAVE THE POPULATION CENTRE ###
            ## THOSE WHO DIE
            # Who is going to die?
            # I suppose the oldest people die (before, some random people died)
            # If I considere that, this while loop
            # can be transformed into a for loop
            
                       
                       
            deaths = 0
            while deaths < population.mortality:
                max_age = 0
                person_to_die = None
                for person in population.inhabitants:
                    if person.age > max_age:
                        max_age = person.age
                        person_to_die = person
                        
                # Update dictionary with ages by range:
                interval = myround(person_to_die.age)
                population.ages_hist[self.year + person_to_die.sex][interval] -= 1
                
                ################ TRYING TO BUILD UP FAMILES ################
                # Remove family
                person_to_die.family.remove_family() # It's working !
                ############################################################
                
                # Remove person
                person_to_die.remove_agent()
                self.remove_person_from_universe(person_to_die)
                deaths += 1
            
            
                
            #print("\n")    
            #print("ESTADO TRAS MUERTES")
            #print("HOMBRES")
            #print(population.ages_hist[self.year + "M"])
            #print("\n")
            #print("MUJERES")
            #print(population.ages_hist[self.year + "F"])
            #print("\n")
            """
            ## SALDO MIGRATORIO (?):
            ## THOSE WHO ARE UNHAPPY ARE GOING TO LEAVE
            #### ¿Y si no hay tantan gente infeliz como gente que se tiene que ir?
            ### Solo hay male ya que los he metido primero en la lista
            ### shuffle of inhabitants list ??????
            
            if population.saldo_migratorio_total < 0:
                saldo = 0
                # Consider each inhabitant
                for person in population.inhabitants:
                    ### THE MOST UNHAPPUY PEOPLE MUST LEAVE !
                    
                    # Is the person unhappy? If so -> remove
                    # But, where is the person going?
                    # (BY NOW) I ASSUME THE PERSON GOES TO A LARGE CITY
                    b = person.migrate()
                    if b and (isinstance(person.family, Fam_one_person)): # CHANGE LATER
                        # I could assume they leave the universe but not
                        #  self.remove_person_from_universe(person)
                        saldo -= 1
                        # Update dictionary with ages by range:
                        interval = myround(person.age)
                        population.ages_hist[self.year + person.sex][interval] -= 1
                        person.remove_agent() # ya esta en person.migrate()
                        
                        # Remove family (origin)
                        person.family.remove_family()
                        
                        # Agent arrives at a large city
                        person.population_centre = random.choice(self.large_cities)
                        
                        # Add agent to destination 
                        person.add_agent()
                        
                        # Add family (destination)
                        person.family.add_family()
                        
                        
                    if saldo == population.saldo_migratorio_total:
                        break
                
        
            ### PEOPLE WHO ARRIVE AT THE POPULATION CENTRE ### 
            
            ## SALDO MIGRATORIO
            ## New guys on the town ! Where are they coming from? 
            ## Dont know, just create new people
            ## CONSIDERING ONLY ONE_PERSON_FAM
            if population.saldo_migratorio_total > 0:
                new_guys = 0
                while new_guys < population.saldo_migratorio_total:
                    # Uodate agent identifiers
                    agent_idx = agent_idx + 1
                    # Create agent
                    the_agent = Agents(identifier = agent_idx,
                                       sex = random.choice(["M", "F"]),
                                       age = random.randrange(25, 101), # Sure?
                                       population_centre = population)
                    
                    # Update family role
                    the_agent.family_role()
                    
                    # Create family for agent
                    my_family = Fam_one_person(population)
                    my_family.update(the_agent)
                    my_family.add_family()
                    
                                       
                    # Add person to the universe
                    self.add_person_to_universe(the_agent)
                    the_agent.add_agent()
                    new_guys += 1
                   
                    # Update dictionary with ages by range:
                    interval = myround(the_agent.age)
                    population.ages_hist[self.year + the_agent.sex][interval] += 1
            
            
            """
            
            ### UPDATE AGES ###
            for person in population.inhabitants:
                if person.family.mig != 1:
                    interval_1 = myround(person.age)
                    person.age += 1
                    person.family_role()
                    interval_2 = myround(person.age)
                    if interval_1 != interval_2:
                        person.population_centre.ages_hist[self.year + person.sex][interval_1] -= 1
                        person.population_centre.ages_hist[self.year + person.sex][interval_2] += 1
                    else:
                        pass
                #else:
                #    person.age += 1
                
            
                    
            
            #################### TRYING TO BUILD UP FAMILIES ##################
            # Time to disband families with kids
            kids_to_adult = 0
            adults_no_kids = 0
            disbanded_fams = 0
            for family in population.families["fam_kids"].copy():
                b = family.disband()
                if not b[0]:
                    kids_to_adult += b[1]
                    adults_no_kids += b[2]
                if b[0]:
                    kids_to_adult += b[1]
                    adults_no_kids += b[2]
                    disbanded_fams += 1
            #print("UPDATE -> DISBANDED FAMILIES with kids: %s" % disbanded_fams)
            #print("UPDATE -> KIDS TO ADULTS              : %s" % kids_to_adult)
            #print("UPDATE -> FREE ADULTS                 : %s" % adults_no_kids)
            #print("\n")
            ###################################################################
            
            
            
            
            ## THOSE WHO ARE NEWBORN BABIES
            # Newborns need a family so we nee dto search for parents
            # Some of them will be assigned to families with previous kids
            # Some of them will be assigned to new parents
            new_borns = 0
            
            #################### TRYING TO BUILD UP FAMILIES ##################
            t0 = 0
            t1 = 0
            t2 = 0
            ####################################################################
            
            while new_borns < population.natality:
                
                agent_idx = agent_idx + 1
                
                # Create agent
                the_agent = Agents(
                            identifier = agent_idx,
                            sex = random.choice(["M", "F"]),
                            age = 0,
                            population_centre = population,
                            population_others = [*self.population_centres, *list(self.large_cities)],
                            mdt               = np.random.triangular(right = population.minmdt,     mode = population.meanmdt,      left = population.maxmdt),
                            carretn           = np.random.triangular(right = population.mincarretn, mode =  population.meancarretn, left = population.maxcarretn),
                            aut               = np.random.triangular(right = population.mindisaut,  mode = population.meandisaut,   left = population.maxdisaut),
                            ferr              = np.random.triangular(right = population.mindisferr, mode = population.meandisferr,  left = population.maxdisferr),
                            dis10m            = np.random.triangular(right = population.mindisn10m, mode = population.meandisn10m,  left = population.maxdisn10m),
                            hospi             = population.disthospit,
                            farma             = population.distfarma,
                            ceduc             = population.distceduc,
                            curgh             = population.distcurgh,
                            atprim            = population.distatprim,
                            salario           = population.salario,
                            gasto             = population.gasto,
                            betas             = self.betas,
                            gamma             = self.gamma,
                            alphas            = self.alphas,
                            theta             = self.theta,
                            )
                
                # Update family role
                the_agent.family_role()
                    
                ## Add agent to the universe
                self.add_person_to_universe(the_agent)
                # Add agent to population centre
                the_agent.add_agent()
                # Update counter
                new_borns += 1
                # Update dictionary with ages by range:
                interval = myround(the_agent.age)
                population.ages_hist[self.year + the_agent.sex][interval] += 1
                
                ################## TRYING TO BUILD UP FAMILIES ################
                # Time to assign kids to families
                
                
                for family in population.families["fam_kids"]:
                    # If available space for kids
                    if len(family.kids) < family.kids_limit:
                        t0 += 1
                        family.update(the_agent, "kid")
                        break
                            
                    elif (family.kids_limit == 1) and (len(family.kids) == 1):
                        t1 += 1
                        family.kids_limit += 1
                        family.update(the_agent, "kid")
                        break
                #if not the_agent.family:
                #    print("FAMILY NOT FOUND FOR KID DICE < THRES")
                        
                    
                if not the_agent.family:
                #else: # Build up a new family   
                    
                    bool_father = False
                    bool_mother = False
                            
                    for family in population.families["fam_one_person"]:
                                                        
                        if family.members.sex == "M":
                            if ((not bool_father) and 
                                (family.members.maybe_parent) and
                                (not family.members.is_kid)):
                                            
                                        
                                if bool_mother:
                                    my_bool = (my_mother.age - 5 <= family.members.age <= my_mother.age + 5)
                                    if my_bool:
                                        my_father = family.members
                                        my_father_family = family
                                        bool_father = True
                                                
                                            
                                else:
                                    if family.members.age in range(random.randint(25, 40)):
                                        my_father = family.members
                                        my_father_family = family
                                        bool_father = True
                                    
                                
                        else: #family2.members.sex == "F":
                            if ((not bool_mother) and 
                                (family.members.maybe_parent) and
                                (not family.members.is_kid)):
                                        
                                        
                                if bool_father:
                                    my_bool = (my_father.age - 5 <= family.members.age <= my_father.age + 5)
                                    if my_bool:
                                        my_mother = family.members
                                        my_mother_family = family
                                        bool_mother = True
                                                       
                                else:
                                    if family.members.age in range(random.randint(25, 40)):
                                        my_mother = family.members
                                        my_mother_family = family
                                        bool_mother = True
                                
                        
                        if (bool_mother) and (bool_father):
                            my_father.family = False
                            my_father_family.remove_family()
                                    
                            my_mother.family = False
                            my_mother_family.remove_family()
                                    
                            my_family = Fam_kids(population, 1)
                            my_family.update(my_father, "father")
                            my_family.update(my_mother, "mother")
                            my_family.update(the_agent, "kid")
                            my_family.add_family()
                                    
                            t2 += 1
                            break
                                
                if not the_agent.family:
                    print("FAMILY NOT FOUND FOR KID")
                    print("\n")
                            
        
                        
            #print("BUG0 %s" % t0)
            #print("BUG1 %s" % t1)
            #print("BUG2 %s" % t2)
                        
                    
                
                ###############################################################
                
                
            for agent in population.inhabitants:
                agent.behavioural_attitude()
                agent.perceived_beahavioural_control()
                agent.subjective_norm()
                agent.intention()
                
                
            
            #print("FAMILIES WITH KIDS")
            
            my_copy = population.families["fam_kids"].copy()
            for family in my_copy:
                if family.mig != 1:
                    if len(family.members) > 0:
                        #print("NEW FAMILY")
                        #print("FAMILY LOCATION %s" % family.population_centre.population_name)
                        dictionaryList = []
                        new = {}
                        for member in family.members:
                            temp = member.intention_hist
                            #print("MEMBER DICT:")
                            #print(temp)
                            #print("\n")
                            dictionaryList.append(temp)
                        for key in temp.keys():
                            new[key] = mean([d[key] for d in dictionaryList ])
                        #print("MEAN")
                        #print(new)
                        #print("\n")
                        #print("Current value for %s (%s) -> %s" %
                        #  (family.population_centre.population_name,
                        #   family.population_centre.population_id,
                        #   str(new[family.population_centre.population_id])))
                    
                    
                        current_pts = new[family.population_centre.population_id]
                    
                        my_origin = family.population_centre.population_id
                    
                        stored_pts  = dict(sorted(new.items(), key=lambda item: item[1], reverse = True))
                        rn = np.random.uniform(0, 1)
                        for key, value in stored_pts.items():
                            if value >= current_pts:
                                #print("Possible migration to %s -> %s" % (str(key), str(value)))
                                if rn >= value:
                                    pass
                                else:
                                    # Migrate
                                    temp = [*self.population_centres, *self.large_cities]
                                    new_population = [x for x in temp if str(x.population_id) == str(key)][0]
                                
                                    attr =[str(x.population_id) for x in self.large_cities]
                                
                                    while family in population.families["fam_kids"]:
                                        family.remove_family()
                                    
                                    
                                    
                                    family.add_family_2(new_population,
                                                    df1 = self.df_income_spend,
                                                    df2 = self.df_income_spend_large_cities,
                                                    year = self.year,
                                                    attr = attr)
                                
                                    with open("pruebas/prueba21/kids.csv", "a", newline = "") as file:
                                        writer = csv.writer(file)
                                        writer.writerow([1, my_origin, key, self.year])
                                        
                                    with open("pruebas/prueba21/total.csv", "a", newline = "") as file:
                                        writer = csv.writer(file)
                                        writer.writerow([len(family.members), my_origin, key, self.year])
                                
                                    #family.population_centre = population
                                    
                                    #for agent_temp in family.members:
                                    #    interval = myround(agent_temp.age)
                                    #    print("ESTAMOS EN EL MUNICIPIO %s" % population.population_name)
                                    #    print(agent_temp.population_centre.population_name)
                                    #    agent_temp.population_centre.ages_hist[self.year + agent_temp.sex][interval] -= 1
                                    #    agent_temp.remove_agent()
                                        
                                    #    if family in population.families["fam_kids"]:
                                    #        family.remove_family()
                                        
                                    #    agent_temp.population_centre = new_population
                                    #    agent_temp.add_agent()
                                        
                                    #    agent_temp.update_infra()
                                    #    agent_temp.update_eco(df_eco_mun = self.df_income_spend,
                                    #                          df_eco_atr = self.df_income_spend_large_cities)
                                    
                                        
                                    #    if new_population not in self.large_cities:
                                    #        agent_temp.population_centre.ages_hist[self.year + agent_temp.sex][interval] += 1
                                        
                                    #    if family not in  new_population.families["fam_kids"]:
                                    #        family.add_family()
                                            
                                     
                                    #with open("Results/kids.csv", "a", newline = "") as file:
                                    #    writer = csv.writer(file)
                                    #    writer.writerow([1, my_origin, key, self.year])
                                        
                                    #with open("Results/total.csv", "a", newline = "") as file:
                                    #    writer = csv.writer(file)
                                    #    writer.writerow([len(family.members), my_origin, key, self.year])
                                    #print("MIGRACION EFECTUADA")
                                    #print("\n")
                                    break
                                    
                                
                            #print("\n")
                            #print("\n")
                    
            
            
            #print("UNIPERSON FAMILIES")
            my_copy_2 = population.families["fam_one_person"].copy()
            for unifamily in my_copy_2:
                if unifamily.mig != 1:
                    #print("NEW UNIFAM %s" % unifamily)
                    #print("FAM LOCATION %s" % unifamily.population_centre.population_name)
                    my_origin = unifamily.population_centre.population_id
                    agent_temp = unifamily.members
                    #print("MEM LOCATION %s" % agent_temp.population_centre.population_name)
                    #print("Current value for %s (%s) -> %s" %
                    #      (agent_temp.population_centre.population_name,
                    #       agent_temp.population_centre.population_id,
                    #       str(agent_temp.intention_hist[agent_temp.population_centre.population_id])))
                    
                    current_pts = agent_temp.intention_hist[agent_temp.population_centre.population_id]
                    stored_pts  = dict(sorted(agent_temp.intention_hist.items(), key=lambda item: item[1], reverse = True))
                    rn = np.random.uniform(0,1)
                    for key, value in stored_pts.items():
                        if value >= current_pts:
                            #print("Possible migration to %s -> %s" % (str(key), str(value)))
                            if rn >= value:
                                pass
                            else:
                                # Migrate 
                                interval = myround(agent_temp.age)
                                unifamily.population_centre.ages_hist[self.year + agent_temp.sex][interval] -= 1
                                agent_temp.remove_agent()
                                unifamily.remove_family()
                                temp = [*self.population_centres, *self.large_cities]
                                new_population = [x for x in temp if str(x.population_id) == str(key)][0]
                                agent_temp.population_centre = new_population
                                agent_temp.add_agent()
                                try:
                                    agent_temp.update_infra()
                                except: 
                                    pass
                                try:
                                    agent_temp.update_eco(df_eco_mun = self.df_income_spend,
                                                      df_eco_atr = self.df_income_spend_large_cities)
                                except:
                                    pass
                            
                                unifamily.add_family()
                                if new_population not in self.large_cities:
                                    agent_temp.population_centre.ages_hist[self.year + agent_temp.sex][interval] += 1
                                
                                
                                with open("pruebas/prueba21/unip.csv", "a", newline = "") as file:
                                        writer = csv.writer(file)
                                        writer.writerow([1, my_origin, key, self.year])
                                    
                                with open("pruebas/prueba21/total.csv", "a", newline = "") as file:
                                        writer = csv.writer(file)
                                        writer.writerow([1, my_origin, key, self.year])
                
                                break
                            #print("\n")
                            #print("\n")
            
            #break
            
            
            ### UPDATE MORTALITY, NATALITY, .... ###
            #d_args_update = {}
            
            for column in self.cols_update:
                if column + self.year in self.df_historic_ages.columns:
                    if column == "NAT":
                        population.natality_real =  self.df_historic_ages.\
                            query('CODMUN == ' + str(population.population_id))[column+self.year].values
                    elif column == "MOR":
                        population.mortality_real =  self.df_historic_ages.\
                            query('CODMUN == ' + str(population.population_id))[column+self.year].values
                    else:
                        population.pob_real =  self.df_historic_ages.\
                            query('CODMUN == ' + str(population.population_id))[column+self.year].values
                            
                else:
                    if column == "NAT":
                        population.natality_real  = None
                    elif column == "MOR":
                        population.mortality_real = None
                    else:
                        population.pob_real       = None
                    
            
            population.update_population()            
                        
            #print("\n")
            #print("FINAL ACTUALIZACION")
            #print("HOMBRES")
            #print(population.ages_hist[self.year + "M"])
            #print("\n")
            #print("MUJERES")
            #print(population.ages_hist[self.year + "F"])
            #print("\n")
            
            
            
            
            #print(population.population_name)
        for population in self.population_centres:
            ### INVOKE NATALITY AND MORTALITY MODELS ###
            df = pd.DataFrame.from_dict(population.ages_hist)        
            my_cols = [col for col in df.columns if str(self.year) in col]
            df = df[my_cols]
            temp = df.transpose()
            temp_male = pd.DataFrame(temp.iloc[0]).transpose()
            temp_male.columns =["H" + x.replace("-", "A") if "-" in x else "H100MS" for x in temp_male.columns]
            temp_female = pd.DataFrame(temp.iloc[1]).transpose()
            temp_female.columns = ["M" + x.replace("-", "A") if "-" in x else "M100YMS" for x in temp_female.columns]
            df_X = pd.concat([temp_male.reset_index(drop=True),
                          temp_female.reset_index(drop=True)], axis=1)
                
            df_X_natality = df_X[['H0A4', 'H5A9', 'H10A14', 'H15A19', 'H20A24', 'H25A29', 'H30A34',
       'H35A39', 'H40A44', 'H45A49', 'H50A54', 'H55A59', 'M0A4', 'M5A9',
       'M10A14', 'M15A19', 'M20A24', 'M25A29', 'M30A34', 'M35A39', 'M40A44',
       'M45A49', 'M50A54', 'M55A59']]
            
            df_X_mortality = df_X[['H60A64', 'H65A69', 'H70A74', 'H75A79', 'H80A84', 'H85A89', 'H90A94',
       'H95A99', 'H100MS', 'M60A64', 'M65A69', 'M70A74', 'M75A79', 'M80A84',
       'M85A89', 'M90A94', 'M95A99', 'M100YMS']]
    
            mortality = self.mortality_model.predict(df_X_mortality) 
            natality  = self.natality_model.predict(df_X_natality,verbose=None)
            #print("Mortality %f" % int(mortality))
            #print("Natality %f" % int(natality))
            #print("\n")
            
            population.mortality = 0 if mortality <= 0 else int(mortality)
            population.natality  = 0 if natality  <= 0 else int(natality)
            
            # Update year for the population centre
            population.year = int(population.year) + 1
            # Update historial for ages
            population.update_population_hist()
            # Update historial for families
            population.update_families_hist()
            
            
        for population in self.population_centres:
            for family in population.families["fam_kids"]:
                family.mig = 0
                
            for family in population.families["fam_one_person"]:
                family.mig = 0
            
       
       
               
    def remove_person_from_universe(self, agent):
        # METHOD TO REMOVE PEOPLE FROM UNIVERSE (those who die mainly)
        self.universe_persons.remove(agent)
        
        
    def add_person_to_universe(self, agent):
        # METHOD TO ADD PEOPLE TO THE UNIVERSE (newborn babies mainly)
        self.universe_persons.append(agent)    
        
        
        
        
    ###########################################################################
    ###########################################################################  
    #########                   MONITORIZATION                        #########
    ###########################################################################
    ###########################################################################
    
    def plot_population_hist(self, population_code):
        # METHOD FOR PLOTTING POPULATION HISTORIAL IN A
        # SPECIFIED POPULATION CENTRE.

        #population_code = int(input("Please, enter a population code: "))
        
        my_population = False
        for population in self.population_centres:
            if population.population_id == population_code:
                my_population = population
        
        if my_population == False:
            raise Exception("Population centre not found")
        
        
        data  = {"NAT" : my_population.natality_hist,
                 "MOR" : my_population.mortality_hist,
                 "HOM" : my_population.men_hist,
                 "MUJ" : my_population.women_hist,
                  #"SALDOMIG" : my_population.saldo_hist,
                 "YEAR" : my_population.year_hist}
        
        df = pd.DataFrame.from_dict(data)
        
        print(df)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x = df["YEAR"], y = np.log(df["HOM"]),
                      mode = "lines",
                      name = "Hombres"))
        
        fig.add_trace(go.Scatter(x = df["YEAR"], y = np.log(df["MUJ"]),
                      mode = "lines",
                      name = "Mujeres"))
        
        fig.add_trace(go.Scatter(x = df["YEAR"], y = np.log(df["NAT"]),
                      mode = "lines",
                      name = "Natalidad"))
        
        fig.add_trace(go.Scatter(x = df["YEAR"], y = np.log(df["MOR"]),
                      mode = "lines",
                      name = "Mortalidad"))
        
        # SALDO MIGRATORIO NEGATIVO -> ERROR !!!
        #fig.add_trace(go.Scatter(x = df["YEAR"], y = np.log(df["SALDOMIG"]),
        #              mode = "lines",
        #              name = "Saldo migratorio"))
        
        fig.update_layout(title = "Evolución de variables en %s" % my_population.population_name,
                    xaxis_title = "Año",
                    yaxis_title = "Total personas (log-scale)")
  
        
        #fig.show()
        return fig
        
    
    
    def plot_population_pyramid_2(self, population_code, year):
        # I guess the plot for population pyramids was a mess
        # Too many pyramids for a tiny window
        # So I thik is better to show each pyramid individually
        
        my_population = False
        for population in self.population_centres:
            if population.population_id == population_code:
                my_population = population
        
        if my_population == False:
            raise Exception("Population centre not found")
        
        
        
        df = pd.DataFrame.from_dict(my_population.ages_hist)        
        my_cols = [col for col in df.columns if str(year) in col]
        df = df[my_cols]

        fig = go.Figure()
        
        fig.add_trace(go.Bar(
                          y = df.index.values.tolist(),
                          x = - df.iloc[:, 0],
                          name  = "Hombres",
                          showlegend = False,
                          marker_color = "blue",
                          orientation = "h",))
            
            
        fig.add_trace(go.Bar(
                          y = df.index.values.tolist(),
                          x =  df.iloc[:, 1],
                          name  = "Mujeres",
                          marker_color = "orange",
                          showlegend = False,
                          orientation = "h",))
        
        fig.add_annotation(x = int(np.min(- df.iloc[:, 0])),
                           y = 20,
                           text = "<b>Hombres<b>",
                           showarrow = False,
                           font = {"size":15})

        fig.add_annotation(x = int(np.max(df.iloc[:, 1])),
                           y = 20,
                           text = "<b>Mujeres<b>",
                           showarrow = False,
                           font = {"size":15})
            
        fig.update_layout(barmode = 'relative',
                               bargap = 0.0, bargroupgap = 0)
        fig.update_xaxes(tickangle = 90)
        
        fig.update_layout(
                    title_text="Pirámide poblacional de %s en %s " 
                        % (my_population.population_name, year))
        #fig.show()
        return fig
        
       
        
    def plot_population_pyramid(self, population_code):
        # METHOD FOR PLOTTING POPULATION HISTORIAL IN A
        # SPECIFIED POPULATION CENTRE. ALDO PLOTS POPULATION PYRAMID

        #population_code = int(input("Please, enter a population code: "))
        
        my_population = False
        for population in self.population_centres:
            if population.population_id == population_code:
                my_population = population
        
        if my_population == False:
            raise Exception("Population centre not found")
        
        
        df = pd.DataFrame.from_dict(my_population.ages_hist)
        
        
        fig = make_subplots(rows = int(len(df.columns) / 2), cols = 1,
                 subplot_titles = np.unique([x[:-1] for x in df.columns]))
        
        
        # Function 2Z -> Z ... i guess not
        row = 1
        for i in range(0, len(df.columns), 2):
            if i == 0:
                show = True
            else:
                show = False
        

            fig.add_trace(go.Bar(
                          y = df.index.values.tolist(),
                          x = - df.iloc[:, i],
                          name  = "Hombres",
                          marker_color = "blue",
                          orientation = "h",
                          showlegend = show),
                          row = row, col = 1)
            
            fig.add_trace(go.Bar(
                          y = df.index.values.tolist(),
                          x =  df.iloc[:, i + 1],
                          name  = "Mujeres",
                          marker_color = "orange",
                          orientation = "h",
                          showlegend = show),
                          row = row, col = 1)
            
            fig.update_layout(barmode = 'relative',
                               bargap = 0.0, bargroupgap = 0)
            fig.update_xaxes(tickangle = 90)


            #for t in fig2.data:
            #    fig.append_trace(t , row = row, col = 1)
                
            row += 1
            
        
        fig.update_layout(
                    title_text="Evolución de la pirámide poblacional en %s" 
                        % my_population.population_name,
                    bargap = 0.0, bargroupgap = 0,)
        #fig.show()
        return fig
    
    
    
    def plot_in_out(self, population_code):
        # Plot in/out migration
                
        my_population = False
        for population in self.population_centres:
            if population.population_id == population_code:
                my_population = population
        
        if my_population == False:
            raise Exception("Population centre not found")
        
        df_temp = self.df_historic_ages.\
                query('CODMUN == ' + str(population.population_id))
                
        years = population.year_hist
        my_in  = ""
        my_out = ""
        for year in years:
            my_out = my_out  + ".*BAJASTT.*" + str(year) +  "|" 
            my_in  = my_in   + ".*ALTASTT.*" + str(year) +  "|"
        
        
        in_val  = df_temp.filter(regex = my_in[:-1],  axis = 1)
        out_val = df_temp.filter(regex = my_out[:-1], axis = 1)
        
        balance = [x-y for (x,y) in zip(in_val.values, out_val.values)]        
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(x = years,
                             y = list(*chain(in_val.values.tolist())),
                             name = "Altas",
                             marker_color = "violet"))
        
        out_val = list(map(lambda x: -x, list(*chain(out_val.values.tolist()))))
        fig.add_trace(go.Bar(x = years,
                             y = out_val,
                             name = "Bajas",
                             marker_color = "yellow"))
        
        fig.add_trace(go.Scatter(x = years,
                                 y = list(*chain(balance)),
                                 name = "Balance",
                                 marker_color = "green"))
        
        fig.update_layout(barmode = 'overlay',
                          xaxis_tickangle = -45,
                          title = "Altas y bajas en %s" %  my_population.population_name)
        
        return fig
        
        
        
        
        
        
    
    def plot_families(self, population_code):
        
        my_population = False
        for population in self.population_centres:
            if population.population_id == population_code:
                my_population = population
        
        if my_population == False:
            raise Exception("Population centre not found")
            
        df = pd.DataFrame.from_dict(my_population.families_hist).transpose()
        df = df.iloc[::-1]
        l = [x + y for (x,y) in zip(my_population.men_hist, my_population.women_hist)]
        l.reverse()
        
        # Creating two subplots
        fig = make_subplots(rows = 1, cols = 2, specs = [[{}, {}]],
                 shared_xaxes = True, shared_yaxes = False,
                 vertical_spacing = 0.002) 
        
        fig.add_trace(go.Bar(
                y = list(df.index.astype(str)),
                x = list(df["num_fam_one_person"]),
                name = 'Familias unipersonales',
                orientation='h',
                marker=dict(
                        color = 'rgba(246, 78, 139, 0.6)',
                        line = dict(color = 'rgba(246, 78, 139, 1.0)', width = 3)
                )
        ), 1, 1)

        fig.add_trace(go.Bar(
                y = list(df.index.astype(str)),
                x = list(df["num_fam_kids"]),
                name = 'Familias con hijos',
                orientation = 'h',
                marker = dict(
                        color = 'rgba(58, 71, 80, 0.6)',
                        line = dict(color = 'rgba(58, 71, 80, 1.0)', width = 3)
                )
        ), 1, 1)

        fig.update_layout(barmode='stack')

        fig.add_trace(go.Scatter(
                x = l,
                y = list(df.index.astype(str)),
                mode = 'lines+markers',
                line_color = 'rgb(128, 0, 128)',
                name = 'Población total',
        ), 1, 2)
        
        fig.update_layout(
                title = 'Evolución del número de familias y población en %s'
                        % my_population.population_name,
                yaxis = dict(
                        showgrid = False,
                        showline = False,
                        showticklabels = True,
                        domain = [0, 0.85],
                ),
            
                yaxis2 = dict(
                        showgrid = False,
                        showline = True,  
                        showticklabels = False,
                        linecolor = 'rgba(102, 102, 102, 0.8)',
                        linewidth = 2,
                        domain = [0, 0.85],
                ),  
                
                xaxis = dict(
                        zeroline = False,
                        showline = False,
                        showticklabels = True,
                        showgrid = True,
                        domain = [0, 0.42],
                ),

                xaxis2 = dict(
                        zeroline = False,
                        showline = False,
                        showticklabels = True,
                        showgrid = True,
                        domain = [0.47, 1],
                        side = 'top',
                        dtick = int(round((max(l) - min(l))/5)),
                ),
                        
                legend = dict(x = 0.029, y = 1.038, font_size = 12),
                margin = dict(l = 100, r = 20, t = 70, b = 70),
                paper_bgcolor = 'rgb(248, 248, 255)',
                plot_bgcolor  = 'rgb(248, 248, 255)',
        )   

        return fig

        
        
    def plot_test_vegetativo(self, population_code):
       
        my_population = False
        for population in self.population_centres:
            if population.population_id == population_code:
                my_population = population
        
        if my_population == False:
            raise Exception("Population centre not found")
        
        
        data  = {"NAT"  : my_population.natality_hist,
                 "MOR"  : my_population.mortality_hist,
                 "POB"  : my_population.population_hist,
                 "POB_REAL"  : my_population.population_hist_real,
                 "MOR_REAL"  : my_population.mortality_hist_real,
                 "NAT_REAL"  : my_population.natality_hist_real,
                 "POB_REAL_SVG_MIGR" : my_population.population_hist_real_migr,
                 "YEAR" : my_population.year_hist}
        
        df = pd.DataFrame.from_dict(data)
        
        df.to_csv("pruebas/prueba21/test_vegetativo"  + str( my_population.population_name).replace(" ", "-") + ".csv",
                  index = False)
        
        
        
        
        
        
        
        fig = make_subplots(
                            rows=2, cols=2,
                            specs=[[{}, {}],
                            [{"colspan": 2}, None]],
                            subplot_titles=("Natalidad","Mortalidad", "Población"))
        
        fig.add_trace(go.Scatter(x = df["YEAR"], y = df["NAT_REAL"],
                      name = "Natalidad observación",
                      showlegend = True,
                      legendgroup = "1",
                      marker = dict(color = "purple")),
                     row = 1, col = 1)
        
        fig.add_trace(go.Scatter(x = df["YEAR"], y = df["NAT"],
                      name = "Natalidad predicción",
                      showlegend = True,
                      legendgroup = "1",
                      marker = dict(color = "pink")),
                     row = 1, col = 1)
        
        fig.add_trace(go.Scatter(x = df["YEAR"], y = df["MOR_REAL"],
                      name = "Mortalidad observación",
                      showlegend = True,
                      legendgroup = "2",
                      marker = dict(color = "green")),
                     row = 1, col = 2)
        
        fig.add_trace(go.Scatter(x = df["YEAR"], y = df["MOR"],
                      name = "Mortalidad predicción",
                      showlegend = True,
                      legendgroup = "2",
                      marker = dict(color = "lightgreen")),
                     row = 1, col = 2)
        
        
        #fig.add_trace(go.Scatter(x = df["YEAR"], y = df["POB_REAL"],
        #              name = "Población observación (SVEG)",
        #              showlegend = True,
        #              legendgroup = "3",
        #              marker = dict(color = "red")),
        #             row = 2, col = 1)
        
        
        fig.add_trace(go.Scatter(x = df["YEAR"], y = df["POB_REAL_SVG_MIGR"],
                      name = "Población observación (SVEG, MIGR)",
                      showlegend = True,
                      legendgroup = "3",
                      marker = dict(color = "red")),
                     row = 2, col = 1)
        
        fig.add_trace(go.Scatter(x = df["YEAR"], y = df["POB"],
                      name = "Población predicción",
                      showlegend = True,
                      legendgroup = "3",
                      marker = dict(color = "orange")),
                      row = 2, col = 1)
        
        
        fig.update_yaxes(rangemode="tozero")
        
        fig.update_layout(
                    title_text ="TEST EVOLUCIÓN en %s" % my_population.population_name 
)
  
        
        return fig
    
    
    def plot_vegetativo_test_pyramid(self, population_code):
        
        year = 2021
        
        my_population = False
        for population in self.population_centres:
            if population.population_id == population_code:
                my_population = population
        
        if my_population == False:
            raise Exception("Population centre not found")
        
        
        
        path = "/home/jesus/Escritorio/PIRAMIDE/pyramid.csv"
        pyramid = pd.read_csv(path)
        if population_code not in pyramid["CODMUN"].values:
            raise Exception("POPULATION CENTRE WITH CODE %s NOT FOUND" % population_code)
        #y_age = pyramid[pyramid["CODMUN"] == population_code]["Rango"]
        x_m = pyramid[pyramid["CODMUN"] == population_code]["Total_HOM"]
        x_f = pyramid[pyramid["CODMUN"] == population_code]["Total_MUJ"]


        
        df = pd.DataFrame.from_dict(my_population.ages_hist)        
        my_cols = [col for col in df.columns if str(year) in col]
        df = df[my_cols]
        
        df.to_csv("pruebas/prueba21/test_piramide"  + str( my_population.population_name).replace(" ", "-") + ".csv")
        
        df_temp = pd.DataFrame({"x_m": list(x_m), "x_f" : list(x_f)})
        df_temp.to_csv("pruebas/prueba21/test_piramide_temp"  + str( my_population.population_name).replace(" ", "-") + ".csv")

       
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
                          y = df.index.values.tolist(),
                          x = - df.iloc[:, 0],
                          name  = "Predicción", #Hombres
                          marker_color = "blue",
                          mode = "lines",
                          showlegend = True))
            
        
        fig.add_trace(go.Scatter(
                        y = df.index.values.tolist(),
                        x = - x_m, 
                        name  = "Observación", #Hombres
                        mode = "lines",
                        showlegend = True,
                        marker=dict(color='red')))
        
        fig.add_trace(go.Bar(
                        y = df.index.values.tolist(),
                        x = - x_m, 
                        name  = "Hombres Observación",
                        orientation = 'h',
                        opacity = 0.5,
                        showlegend = False,
                        marker=dict(color='red')))
            
        fig.add_trace(go.Scatter(
                          y = df.index.values.tolist(),
                          x = df.iloc[:, 1],
                          name  = "Mujeres Predicción",
                          marker_color = "blue",
                          mode = "lines",
                          orientation = "h",
                          showlegend = False))
        
        fig.add_trace(go.Scatter(
                         y = df.index.values.tolist(),
                        x = x_f, 
                        name  = "Mujeres Observación",
                        mode = "lines",
                        showlegend = False,
                        marker=dict(color='red')))
            
        fig.add_trace(go.Bar(
                        y = df.index.values.tolist(),
                        x = x_f, 
                        name  = "Mujeres Observación",
                        orientation = 'h',
                        opacity = 0.5,
                        showlegend = False,
                        marker=dict(color='red')))
        
        
        fig.add_annotation(x = int(np.min(- x_m)),
                           y = 20,
                           text = "<b>Hombres<b>",
                           showarrow = False,
                           font = {"size":15})

        fig.add_annotation(x = int(np.max(x_f)),
                           y = 20,
                           text = "<b>Mujeres<b>",
                           showarrow = False,
                           font = {"size":15})
        
        fig.update_layout(barmode = 'overlay') #overlay)
        
        fig.update_xaxes(tickangle = 90)
        
        fig.update_layout(title_text="Pirámide poblacional de %s en %s " 
                        % (my_population.population_name, year),
                           bargap = 0.0,
                           bargroupgap = 0)
        return fig
        
        

    def plot_behavioural_attitude(self, population_code, year):
       
        my_population = False
        for population in [*self.population_centres, *list(self.large_cities)]:
            if population.population_id == population_code:
                my_population = population
        
        if my_population == False:
            raise Exception("Population centre not found")
            
        df = pd.DataFrame.from_dict(my_population.ba_hist[year])
        
        fig = go.Figure()

        for col in df.columns:
            for population in [*self.population_centres, *list(self.large_cities)]:
                if population.population_id == col:
                    name = population.population_name
            fig.add_trace(go.Box(y=df[col].values, name=name))
            
        fig.update_layout(title_text = "Actitud (BA) en %s hacia el resto de municipios en el año %s" 
                        % (my_population.population_name, str(year)),
                        title_font=dict(size=25))
        
        fig.update_xaxes(tickfont=dict(size=20), tickangle=-45)
        fig.update_yaxes(tickfont=dict(size=17))
        
        fig.update_layout(showlegend=False)
        
        return fig
    
    
    def plot_perceived_behavioural_control(self, population_code, year):
       
        my_population = False
        for population in [*self.population_centres, *list(self.large_cities)]:
            if population.population_id == population_code:
                my_population = population
        
        if my_population == False:
            raise Exception("Population centre not found")
            
        df = pd.DataFrame.from_dict(my_population.pbc_hist[year])
        
        fig = go.Figure()

        for col in df.columns:
            for population in [*self.population_centres, *list(self.large_cities)]:
                if population.population_id == col:
                    name = population.population_name
            fig.add_trace(go.Box(y=df[col].values, name=name))
            
        fig.update_layout(title_text = "Control del comportamieto percibido (PBC) en %s hacia el resto de municipios en el año %s" 
                        % (my_population.population_name, str(year)),
                        title_font=dict(size=25))
        fig.update_layout(showlegend=False)
        
        fig.update_xaxes(tickfont=dict(size=20), tickangle=-45)
        fig.update_yaxes(tickfont=dict(size=17))
  
        return fig
    
    
    
    def plot_subjective_norm(self, population_code, year):
       
        my_population = False
        for population in [*self.population_centres, *list(self.large_cities)]:
            if population.population_id == population_code:
                my_population = population
        
        if my_population == False:
            raise Exception("Population centre not found")
            
        df = pd.DataFrame.from_dict(my_population.sn_hist[year])
        for key in my_population.sn_hist.keys():
            print("Key %s -> %s" % (str(key), str(len(my_population.sn_hist[key]))))
        
        fig = go.Figure()

        for col in df.columns:
            for population in [*self.population_centres, *list(self.large_cities)]:
                if population.population_id == col:
                    name = population.population_name
            fig.add_trace(go.Box(y=df[col].values, name=name))
            
        fig.update_layout(title_text = "Norma Subjetiva (SN) en %s hacia el resto de municipios en el año %s" 
                        % (my_population.population_name, str(year)),
                        title_font=dict(size=25))
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickfont=dict(size=20), tickangle=-45)
        fig.update_yaxes(tickfont=dict(size=17))
  
        return fig
    
    
    def plot_intention(self, population_code, year):
       
        my_population = False
        for population in self.population_centres:
            if population.population_id == population_code:
                my_population = population
        
        
        if my_population == False:
            raise Exception("Population centre not found")
            
    
        df = pd.DataFrame.from_dict(my_population.intention_hist[year])
        
        fig = go.Figure()

        for col in df.columns:
            for population in [*self.population_centres, *list(self.large_cities)]:
                if population.population_id == col:
                    name = population.population_name
            fig.add_trace(go.Box(y=df[col].values, name=name))
            
        fig.update_layout(title_text = "Intención (I) en %s hacia el resto de municipios en el año %s" 
                        % (my_population.population_name, str(year)),
                        title_font=dict(size=25))
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickfont=dict(size=20), tickangle=-45)
        fig.update_yaxes(tickfont=dict(size=17))
  
        return fig
            

    def get_poblacion_por_mun(self):
        aux = np.empty((0,1),int)
        for population in self.population_centres:
            #print("Municipio: " + population.population_name)
            aux = np.append(aux,population.get_poblacion())
        return aux
        
        
    
    def regression_metrics(self):
        print("--- REGRESSION METRICS ---")
        for population in self.population_centres:
            years = population.year_hist
            total_pred = [sum(x) for x in zip(population.men_hist, population.women_hist)]
            total_obs = []
            for year in years:
                temp = self.df_historic_ages.\
                    query('CODMUN == ' + str(population.population_id))["POB"+ str(year)].\
                    values
                temp = int(temp)
                total_obs.append(temp)
        
            
            print("- " + population.population_name.upper() + " -")
            print(total_pred)
            print(total_obs)
            print("Explained variance:  %s" % explained_variance_score(total_pred, total_obs))
            print("MAE:  %s" % mean_absolute_error(total_pred, total_obs))
            print("MSE:  %s" % mean_squared_error(total_pred, total_obs))
            print("R2:  %s" % r2_score(total_pred, total_obs))
            print("\n")
        
        
        
    def Print(self):
        print('###################################################')
        print('###################################################')
        print('#    POPULATION CENTRES IN THE UNIVERSE. ' + self.year +'     #')
        print('###################################################')
        print('###################################################')
        print("Universe population: %s persons" % len(self.universe_persons))
        print("\n")
        for population in self.population_centres:
            population.Print()
            population.Print_features()
            ################### TRYING TO BUILD UP FAMILIES ###################
            print("\n")
            #population.Print_families()
            ###################################################################
        #for city in self.large_cities:
        #    city.Print_features()   
