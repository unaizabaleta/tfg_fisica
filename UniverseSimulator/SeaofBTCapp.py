#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:37:41 2022

@author: jesus
"""

import tkinter as tk
from tkinter import messagebox
import sys
from PIL import ImageTk, Image
from paho.mqtt.client import Client 
from time import sleep

from Universe import Universe
from PopulationCentre import PopulationCentre
from LargeCity import LargeCity
from Agents import Agents
from Family_version_3 import Fam_one_person
from Family_version_3 import Fam_kids

import pandas as pd
import numpy as np
  
          
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from  matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import plotly.offline as py
import sys


LARGE_FONT = ("Verdana", 12)



class Pages:
    name = None
    _id  = None
    year = None

class SeaofBTCapp(Pages, tk.Tk):

    def __init__(self, universe, *args, **kwargs):
               
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side = "top", fill = "both", expand = True)

        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)

        self.universe = universe
        self.frames = {}
        

        for F in (StartPage, PageOne, IterationPage,
                  PopulationCentrePage, PlotPage, YearsPage,
                  YearsPageBA, YearsPageSN, YearsPagePBC, YearsPageI):

            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row = 0, column = 0, sticky="nsew")
    
            
        self.show_frame(StartPage)
        
        
    def show_frame(self, cont):
        
        frame = self.frames[cont]
        frame.tkraise()
        
    def show_frame_set_population(self, cont, name, _id):
        
        frame = self.frames[cont]
        frame.tkraise()
        
        Pages.name = name
        Pages._id = _id
        
        

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        
        global welcome_pic 
        
        tk.Frame.__init__(self,parent)
        
        welcome_pic = ImageTk.PhotoImage(Image.open("bienvenida.jpg"))
        tk.Label(self,
                 image = welcome_pic).pack()
        
        self.configure(background='black')
        
        tk.Button(self,
                  text = "ENTER",
                  command = lambda: controller.show_frame(PageOne)).pack()




class IterationPage(tk.Frame):

    def __init__(self, parent, controller):
        
        
        def iterate_universe(n_iter):
            #  tkinter updates things when your program is idle. !!!!
            label = tk.Label(self,
                     text = "...ITERANDO... \n AÑO %s ALCANZADO \n CONTINÚE" % str(2010 + n_iter),
                     font = ("Times", 14))
            label.place(anchor="center", relx=0.5, rely=0.5)
            
            for i in range(1, n_iter):
                controller.universe = controller.universe.update()
                        
               
        tk.Frame.__init__(self, parent)
                
        
        
        
        iter_label = tk.Label(self,
                              text = "INTRODUZCA NÚMERO DE ITERACIONES",
                              font = ("Times", 20))
        
        
        year_label = tk.Label(self,
                              text = "AÑO DE INICIO: 2010",
                              font = ("Times", 20))
        
        year_label2 = tk.Label(self,
                              text = "1 iteración = 1 año",
                              font = ("Times", 20))
        
        
        iter_entry = tk.Entry(self)
        iter_label.pack()
        year_label.pack()
        year_label2.pack()
        iter_entry.pack()
         
        button_ok = tk.Button(self,
                              text='COMENZAR EJECUCIÓN',
                              command = lambda: iterate_universe(int(iter_entry.get())))
        button_ok.pack()
        
        
        tk.Button(self,
                  text = "CONTINUAR",
                  command = lambda: controller.show_frame(PageOne)).\
                  pack(side = "bottom")
        
        
        
class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        
        global comein_pic
        
        tk.Frame.__init__(self, parent)
        
        tk.Button(self,
                  text = "CONSULTA", 
                  command = lambda: controller.show_frame(PopulationCentrePage)).pack()
        
        tk.Button(self,
                  text = "ATRÁS",
                  command = lambda: controller.show_frame(StartPage)).pack()
        
        
      
class PopulationCentrePage(Pages, tk.Frame,):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        #var = tk.StringVar()

        def selection():
            towns_names = []
            cname = lb.curselection()
            for i in cname:
                op = lb.get(i)
                towns_names.append(op)
            
            towns_object = []
            for town in towns_names:
                for population in controller.universe.population_centres:
                    if town == population.population_name:
                        towns_object.append(population)
            
            return [towns_names, towns_object]
        
        
        def showSelected():
            towns = selection()
            for item in towns:
                print(item)
            
            
        show = tk.Label(self, text = "SELECCIONE UN MUNICIPIO", font = ("Times", 28), padx = 10, pady = 10)
        show.pack() 
        lb = tk.Listbox(self, selectmode = "multiple")
        lb.pack(padx = 10, pady = 10, fill = "both") 
        
        
        for item in range(len(controller.universe.population_centres)): 
            lb.insert("end", controller.universe.population_centres[item].population_name) 
            lb.itemconfig(item, bg="#bdc1d6") 

        tk.Button(self,
                  text = "CONFIRMAR CONSULTA",
                  command = showSelected).pack()
        
        tk.Button(self,
                  text = "SIGUIENTE",
                  command= lambda : controller.show_frame_set_population(PlotPage, name = selection()[0][0], _id = str(selection()[1][0].population_id))).pack()
        
        tk.Button(self,
                  text = "ATRÁS",
                  command = lambda : controller.show_frame(PageOne)).pack()
    


class YearsPage(Pages, tk.Frame):
    
    def __init__(self, parent, controller):
        
        def plotter(year):
            fig = controller.universe.plot_population_pyramid_2(int(Pages._id), year)    
            py.plot(fig, filename = "piramide.html", auto_open = True)
        
        tk.Frame.__init__(self, parent)
                
        #_id = int(Pages._id)
        #for item in range(len(controller.universe.population_centres)):
        #    if item.population_id == _id:
        #        my_population = item
        
        years = list(controller.universe.population_centres[0].year_hist)
        
        for year in years:
            tk.Button(self,
                  text = "AÑO %s" % year, 
                  command = lambda year = year: plotter(int(year))).pack()
                  
        tk.Button(self,
                  text = "ATRÁS",
                  command = lambda: controller.show_frame(PlotPage)).pack()


class YearsPageBA(Pages, tk.Frame):
    
    def __init__(self, parent, controller):
        
        def plotter(year):
            path = "pruebas/prueba21/"
            fig = controller.universe.plot_behavioural_attitude(int(Pages._id), year)    
            py.plot(fig, filename = path + "tpb_ba"+ str(Pages.name)+".html", auto_open = True)
        
        tk.Frame.__init__(self, parent)
        
        years = list(controller.universe.population_centres[0].year_hist[1:])
        
        for year in years:
            tk.Button(self,
                  text = "AÑO %s" % year, 
                  command = lambda year = year: plotter(int(year))).pack()
                  
        tk.Button(self,
                  text = "ATRÁS",
                  command = lambda: controller.show_frame(PlotPage)).pack()
        


class YearsPageSN(Pages, tk.Frame):
    
    def __init__(self, parent, controller):
        
        def plotter(year):
            path = "pruebas/prueba21/"
            fig = controller.universe.plot_subjective_norm(int(Pages._id), year)    
            py.plot(fig, filename = path + "tpb_pbc"+ str(Pages.name)+".html", auto_open = True)
        
        tk.Frame.__init__(self, parent)
        
        years = list(controller.universe.population_centres[0].year_hist[1:])
        
        for year in years:
            tk.Button(self,
                  text = "AÑO %s" % year, 
                  command = lambda year = year: plotter(int(year))).pack()
                  
        tk.Button(self,
                  text = "ATRÁS",
                  command = lambda: controller.show_frame(PlotPage)).pack()
        
        
class YearsPagePBC(Pages, tk.Frame):
    
    def __init__(self, parent, controller):
        
        def plotter(year):
            path = "pruebas/prueba21/"
            fig = controller.universe.plot_perceived_behavioural_control(int(Pages._id), year)    
            py.plot(fig, filename = path + "tpb_pbc"+ str(Pages.name)+".html", auto_open = True)
        
        tk.Frame.__init__(self, parent)
        
        years = list(controller.universe.population_centres[0].year_hist[1:])
        
        for year in years:
            tk.Button(self,
                  text = "AÑO %s" % year, 
                  command = lambda year = year: plotter(int(year))).pack()
                  
        tk.Button(self,
                  text = "ATRÁS",
                  command = lambda: controller.show_frame(PlotPage)).pack()



class YearsPageI(Pages, tk.Frame):
    
    def __init__(self, parent, controller):
        
        def plotter(year):
            path = "pruebas/prueba21/"
            fig = controller.universe.plot_intention(int(Pages._id), year)    
            py.plot(fig, filename = path + "tpb_intention_" + str(Pages.name) + ".html", auto_open = True)
        
        tk.Frame.__init__(self, parent)
        
        years = list(controller.universe.population_centres[0].year_hist[1:])
        
        for year in years:
            tk.Button(self,
                  text = "AÑO %s" % year, 
                  command = lambda year = year: plotter(int(year))).pack()
                  
        tk.Button(self,
                  text = "ATRÁS",
                  command = lambda: controller.show_frame(PlotPage)).pack()
        
        
        
class PlotPage(Pages, tk.Frame,):
    
    def __init__(self, parent, controller):
        
        def button_1_plot():
            path   = "pruebas/prueba21/"
            fig = controller.universe.plot_population_hist(int(Pages._id))    
            py.plot(fig, filename = path + str(Pages.name).replace(" ", "-") +"multiline.html", auto_open = True)
            
        def button_2_plot():
            path   = "pruebas/prueba21/"
            fig = controller.universe.plot_population_pyramid(int(Pages._id))    
            py.plot(fig, filename = path + str(Pages.name).replace(" ", "-") +"piramide_2.html", auto_open = True)
            
        def button_3_plot():
            path   = "pruebas/prueba21/"
            fig = controller.universe.plot_families(int(Pages._id))    
            py.plot(fig, filename = path + str(Pages.name).replace(" ", "-") +"piramide_3.html", auto_open = True)
                        
        def button_4_plot():
            path   = "pruebas/prueba21/"
            fig = controller.universe.plot_in_out(int(Pages._id))    
            py.plot(fig, filename = path + str(Pages.name).replace(" ", "-") +"in_out.html", auto_open = True)
            
        def button_5_plot():
            path   = "pruebas/prueba21/"
            fig = controller.universe.plot_test_vegetativo(int(Pages._id))    
            py.plot(fig, filename = path + str(Pages.name).replace(" ", "-") + "-test_vegetativo.html", auto_open = True)
            
        def button_6_plot():
            path   = "pruebas/prueba21/"
            fig = controller.universe.plot_vegetativo_test_pyramid(int(Pages._id))    
            py.plot(fig, filename = path + str(Pages.name).replace(" ", "-") + "-piramide_6.html", auto_open = True)
            
        def button_7_plot():
            path   = "pruebas/prueba21/"
            fig = controller.universe.plot_behavioural_attitude(int(Pages._id))    
            py.plot(fig, filename = path + str(Pages.name).replace(" ", "-") +"tpb_ba.html", auto_open = True)
            
        def button_8_plot():
            path   = "pruebas/prueba21/"
            fig = controller.universe.plot_subjective_norm(int(Pages._id))    
            py.plot(fig, filename = path + str(Pages.name).replace(" ", "-") +"tpb_pbc.html", auto_open = True)
            
        def button_9_plot():
            path   = "pruebas/prueba21/"
            fig = controller.universe.plot_perceived_behavioural_control(int(Pages._id))    
            py.plot(fig, filename = path + str(Pages.name).replace(" ", "-") +"tpb_pbc.html", auto_open = True)
            
        def button_10_plot():
            path   = "pruebas/prueba21/"
            fig = controller.universe.plot_intention(int(Pages._id))    
            py.plot(fig, filename = path + str(Pages.name).replace(" ", "-") +"tpb_i.html", auto_open = True)
            
        ## All of these files mut de removed
        
        global comein_pic
        
        tk.Frame.__init__(self, parent)
        
        tk.Button(self,
                  text = "CONSULTA ACUTAL",
                  command = self.confirm_query).pack()
        
        
        self.label = tk.Label(self, text = "SELECCIONE TIPO DE GRÁFICO").pack()
        
        tk.Button(self,
                  text = "GRÁFICO MULTILINEA: EVOLUCIÓN TEMPORAL",
                  command = button_1_plot).pack()
        
        tk.Button(self,
                  text="PIRÁMIDE POBLACIONAL: EVOLUCIÓN TEMPORAL",
                  command = lambda: controller.show_frame(YearsPage)).pack()
                  # In case we want all the population pyramids shown in the same window
                  #command = lambda: button_2_plot).pack()
        
        tk.Button(self,
                  text="FAMILIAS: EVOLUCIÓN TEMPORAL",
                  command = button_3_plot).pack()
        
        tk.Button(self,
                  text="MIGRACIONES: IN-OUT",
                  command = button_4_plot).pack()
        
        tk.Button(self,
                  text = "TEST - VEGETATIVO",
                  command = button_5_plot).pack()
        
        tk.Button(self,
                  text = "TEST - VEGETATIVO - PIRAMIDE",
                  command = button_6_plot).pack()
        
        tk.Button(self,
                  text = "TPB - Behavioural Attitude (BA)",
                  command = lambda: controller.show_frame(YearsPageBA)).pack()       
        
        tk.Button(self,
                  text = "TPB - Social Norm (SN)",
                  command = lambda: controller.show_frame(YearsPageSN)).pack() 
        
        tk.Button(self,
                  text = "TPB - Perceived Behavioural Control (PBC)",
                  command = lambda: controller.show_frame(YearsPagePBC)).pack()
        
        tk.Button(self,
                  text = "TPB - Intention (I)",
                  command = lambda: controller.show_frame(YearsPageI)).pack()
        
        tk.Button(self,
                  text = "ATRÁS",
                  command = lambda: controller.show_frame(PopulationCentrePage)).pack()
        
        button_destroy = tk.Button(self,
                                   text = "CERRAR",
                                   command = lambda: controller.destroy())
        
        button_destroy.pack(side = "bottom")
        
     
    def confirm_query(self):
        text = str("HA ESCOGIDO EL MUNICIPIO %s CON CÓDIGO %s" % (Pages.name, Pages._id))
        self.label = tk.Label(self, text = text).pack()

        
        
        
        
        


