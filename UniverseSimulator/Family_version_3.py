#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 10:35:29 2022

@author: jesus
"""
import math


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


import warnings
warnings.simplefilter("always")

class Family():
    
    def __init__(self):
        pass
        
        
    def add_family(self):
        if isinstance(self, Fam_one_person):
    
            # POPULATION_CENTRE ATTRIBUTE OF FAMILIES MUST BE UPDATED 
            # AT THE SAME TIME AS THE POPULATION_CENTRE ATTRIBUTE FOR AGENTS
            self.population_centre = self.members.population_centre
            self.population_centre.families["fam_one_person"].append(self)
            
        elif isinstance(self, Fam_kids):
            self.population_centre = self.members[0].population_centre
            self.population_centre.families["fam_kids"].append(self)
        
        
        else:
            warnings.warn("FAMILY CLASS UNDEFINED")
            
    
    def add_family_2(self, population, df1, df2, year ,attr):
        self.population_centre = population
        self.population_centre.families["fam_kids"].append(self)
        for agent in self.members:
            interval = myround(agent.age)
            if str(agent.population_centre.population_id) not in attr:
                agent.population_centre.ages_hist[year + agent.sex][interval] -= 1
            agent.remove_agent()
            agent.population_centre = population
            agent.add_agent()
            try:
                agent.update_infra()
            except:
                pass
            try:
                agent.update_eco(df_eco_mun = df1,
                                 df_eco_atr = df2)
            except:
                pass
            
            if str(agent.population_centre.population_id) not in attr:
                agent.population_centre.ages_hist[year + agent.sex][interval] += 1
            
        
    
    def remove_family(self):
        if isinstance(self, Fam_one_person):
            self.population_centre.families["fam_one_person"].remove(self)
        elif isinstance(self, Fam_kids):
            self.population_centre.families["fam_kids"].remove(self)
        else:
            warnings.warn("UNABLE TO REMOVE FAMILY")
            
        
        
        
class Fam_one_person(Family):
    
    def __init__(self, population_centre):
        self.mig = 0
        self.members = None
        self.population_centre = population_centre
        
    def update(self, agent):
        if agent.family:
            warnings.warn("THIS AGENT ALREADY HAS A FAMILY!")
        if self.members:
            warnings.warn("THIS FAMILY IS COMPLETE")
        else:
            self.members = agent
            #agent.family = True
            agent.family = self
    
            
class Fam_kids(Family):
    
    def __init__(self, population_centre, kids_limit):
        self.mig = 0
        self.population_centre = population_centre
        self.kids_limit = kids_limit
        self.members = []
        self.father = None
        self.mother = None
        self.kids = []
        
    def update(self, agent, role):
        if agent.family:
            warnings.warn("THIS AGENT  ALREADY HAS A FAMILY!")
        if role == "father":
            if self.father:
                warnings.warn("THIS FAMILY ALREADY HAS A FATHER")
            else:
                self.father = agent
                self.members.append(agent)
                #agent.family = True
                agent.family = self
        elif role == "mother":
            if self.mother:
                warnings.warn("THIS FAMILY ALREADY HAS A MOTHER")
            else:
                self.mother = agent
                self.members.append(agent)
                #agent.family = True
                agent.family = self
        elif role == "kid":
            if len(self.kids) >= self.kids_limit:
                warnings.warn("ENOUGH KIDS")
            else:
                self.kids.append(agent)
                self.members.append(agent)
                #agent.family = True
                agent.family = self
        else:
            warnings.warn("UNAVAILABLE ROLE")
        
        
        
      
    # Break up of a family when all of the kids are older then 25 years old
    def disband(self):
        # Comnsider each kid
        mem_kid = 0
        mem_parent = 0
        for kid in self.kids.copy():
            if kid.age >= 18:
                self.kids.remove(kid)
                kid.family = False
                my_family = Fam_one_person(kid.population_centre)
                my_family.update(kid)
                my_family.add_family()
                mem_kid += 1
            
        # no more kids in the family -> free agents
        if not self.kids: 
            
            # remove family
            #self.population_centre.families["fam_kids"].remove(self)
            self.remove_family()
            
            # new one person family for the father
            if self.father is not None:
                self.father.family = False
                my_family = Fam_one_person(self.father.population_centre)
                my_family.update(self.father)
                # add fmaily to population centre
                my_family.add_family()
            
            # new one persone family for the mother
            if self.mother is not None:
                self.mother.family = False
                my_family = Fam_one_person(self.mother.population_centre)
                my_family.update(self.mother)
                # add family to population centre
                my_family.add_family()

                mem_parent += 2
            
            return [True, mem_kid, mem_parent]
            
        # the family still has kids
        # do not remove family
        else:
            return [False, mem_kid, mem_parent]
            
       
   