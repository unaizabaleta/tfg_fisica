#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:09:49 2022

@author: jesus
"""

import random

import warnings
warnings.simplefilter('always')

class Family():
    
    def __init__(self, population_centre):
        self.population_centre = population_centre


class Fam_unipersonal(Family):
    
    def __init__(self, population_centre):
        self.members = []
    
    def update(self, agent):
        if agent.family:
            warnings.warn("THIS AGENT ALREADY HAS A FAMILY")
        if len(self.members) > 1:
            warnings.warn("FAMILIA UNIPERSONAL. Ya hay una persona")
        else:
            self.members.append(agent)
            agent.family = True
        
class Fam_monoparental(Family):
    
    def __init__(self, population_centre, kids_limit):
        self.kids_limit = kids_limit
        self.members = []
        self.parent = []
        self.kids   = []
    
    def update(self, agent, rol):
        if agent.family:
            warnings.warn("THIS AGENT ALREADY HAS A FAMILY")
        if rol == "parent":
            if len(self.parent) > 1:
                warnings.warn("FAMILIA MONOPARENTAL. Ya hay progenitor")
            else:
                self.parent.append(agent)
                agent.family = True
        if rol == "kid":
            if len(self.kids) >= self.kids_limit:
                warnings.warn("FAMILIA MONOPARENTAL. Suficientes ninios")
            else:
                self.kids.append(agent)
                agent.family = True
            
        
class Fam_pareja_no_ninios(Family):
    
    def __init__(self, population_centre):
        self.members = []
        self.boy  = []
        self.girl = []
    
    def update(self, agent):
        if agent.family:
            warnings.warn("THIS AGENT ALREADY HAS A FAMILY")
        if agent.sex == "M":
            if len(self.boy) >= 1:
                warnings.warn("PAREJA CLASICA SIN HIJOS. Ya hay hombre")
            else:
                self.boy.append(agent)
                self.members.append(agent)
                agent.family = True
        else:
            if len(self.girl) >= 1:
                warnings.warn("PAREJA CLASICA SIN HIJOS. Ya hay mujer")
            else:
                self.girl.append(agent)
                self.members.append(agent)
                agent.family = True
 
        
class Fam_ninios(Family):
    
    def __init__(self, population_centre, kids_limit):
        self.kids_limit = kids_limit
        self.members = []
        self.father = []
        self.mother = []
        self.kids = []

        
    def update(self, agent, rol):
        if agent.family:
            warnings.warn("THIS AGENT ALREADY HAS A FAMILY")
        if rol == "father":
            if len(self.father) >= 1:
                warnings.warn("FAMILIA NINIOS CLASICA: ya hay padre")
            else:
                self.father.append(agent)
                self.members.append(agent)
                agent.family = True
        if rol == "mother":
            if len(self.mother) >= 1:
                warnings.warn("FAMILIA NINIOS CLASICA: ya hay madre")
            else:
                self.mother.append(agent)
                self.members.append(agent)
                agent.family = True
        if rol == "kid":
            if len(self.kids) >= self.kids_limit:
                warnings.warn("FAMILIA MONOPARENTAL. Suficientes ninios")
            else:
                self.kids.append(agent)
                self.members.append(agent)
                agent.family = True
                #warnings.warn("FAMILIA NINIOS CLASICA: Hijo numero %s" % len(self.kids))
        

class Fam_otros(Family):
    
    def __init__(self, population_centre, limit = 100):
        self.members = []
        self.limit = limit
    
    def update(self, agent):
        if agent.family:
            warnings.warn("THIS AGENT ALREADY HAS A FAMILY")
        if len(self.members) >= self.limit:
            warnings.war("FAMILIA OTRO: Limite de gente superado")
        else:
            self.members.append(agent)
            agent.family = True
            #warnings.warn("FAMILIA OTRO: Total gente %s" % len(self.members))
            
            
class Fam_centros(Family):
    def __init__(self, population_centre, capacity = 100):
        self.members = []
        self.capacity = capacity
        
        
    def update(self, agent):
        if agent.family:
            warnings.warn("THIS AGENT ALREADY HAS A FAMILY")
        else:
            pass
        if len(self.members) >= self.capacity:
            warnings.war("FAMILIA OTRO: capacidad de gente superado")
        else:
            self.members.append(agent)
            agent.family = True
    
        

    
    
        