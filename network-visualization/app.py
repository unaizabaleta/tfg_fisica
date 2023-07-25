#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:42:06 2022
@author: jesus
"""


import dash
from dash import dcc
from dash import html
import networkx as nx
import plotly.graph_objs as go

import pandas as pd
from colour import Color
from datetime import datetime
from textwrap import dedent as d
import json


import webbrowser
from threading import Timer

# Import the css template, and pass the css template into dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Migration Network"

# "Default" parameters
PERIOD = [2010, 2019]
POPULATION_CENTRE = "A"


def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")
    

def network_graph(period, population_centre):

    # Read csv for edges
    edges = pd.read_csv('edges.csv')
    # Convert Source and Target column to str for future button
    edges["Source"] = edges["Source"].map(str)
    edges["Target"] = edges["Target"].map(str)
    
    # Read csv for nodes
    nodes = pd.read_csv('nodes.csv')
    # Convert CODMUN oclumn to str for futre button
    nodes["CODMUN"] = nodes["CODMUN"].map(str)

    # Filter record by datetime
    # Add empty Datetime column to edges dataframe
    edges['Datetime'] = "" 
    # Set for unique population centres (CODMUN)
    accountSet = set() 
    # Consider each edge (dataframe row)
    for index in range(0,len(edges)):
        # Proper format for dates
        edges['Datetime'][index] = datetime.strptime(str(edges['Date'][index]), '%Y')
        # Select subset according to datas
        if ((edges['Datetime'][index].year < period[0]) or
            (edges['Datetime'][index].year > period[1])):
            # If out of range -> do not consider the edge
            edges.drop(axis = 0, index = index, inplace = True)
            continue
        # Add population centre to set
        accountSet.add(edges['Source'][index])
        accountSet.add(edges['Target'][index])


    #names = list(accountSet)
    #accountSet = [node1[node1["CODMUN"] == x]["Nombre"].values[0] for x in names]
    
    # Define the centric point of the networkx layout
    shells=[]
    shell1=[]
    shell1.append(population_centre)
    shells.append(shell1)
    shell2=[]
    for elem in accountSet:
        if elem != population_centre:
            shell2.append(elem)
    shells.append(shell2)

    # Create graph using networkx form pandas df
    G = nx.from_pandas_edgelist(edges, # edge list representation
                                source = 'Source', # column for source nodes
                                target = 'Target', # column for target nodes
                                edge_attr = ['Source', 'Target',
                                             'TransactionAmt', 'Date'], # edge attributes
                                create_using = nx.MultiDiGraph()) # Graph type to create
    
    # Add node attributes
    nx.set_node_attributes(G = G, # NetworkX Graph
                           values = nodes.set_index('CODMUN')['Nombre'].to_dict(), # What the nose attribute should be set to
                           name = 'Nombre') # Name of the node attribute
    
    # Add node attributes
    nx.set_node_attributes(G = G, # NetworkX Graph
                           values = nodes.set_index('CODMUN')['Type'].to_dict(), # What the nose attribute should be set to
                           name = 'Tipo')  # Name of the node attribute
    
        
    # CUSTOMIZE FOR BEST LAYOUT    
    # pos = nx.layout.spring_layout(G)
    # pos = nx.layout.circular_layout(G)
    # nx.layout.shell_layout only works for more than 3 nodes
    
    if len(shell2) > 1:
        pos = nx.drawing.layout.shell_layout(G, shells)
    else:
        pos = nx.drawing.layout.spring_layout(G)
    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])


    #  If no node in shell 2
    if len(shell2) == 0:
        traceRecode = []  
        # Scatter points
        node_trace = go.Scatter(x = tuple([1]), #
                                y = tuple([1]),
                                text = tuple([str(population_centre)]),
                                textposition = "bottom center",
                                mode = 'markers+text',
                                marker = {'size': 50, 'color': 'LightSkyBlue'})
        traceRecode.append(node_trace)

        node_trace1 = go.Scatter(x = tuple([1]),
                                 y = tuple([1]),
                                 mode = 'markers',
                                 marker = {'size': 50, 'color': 'LightSkyBlue'},
                                 opacity = 0)
        traceRecode.append(node_trace1)

        # figure Settings
        figure = {
            "data": traceRecode,
            "layout": go.Layout(title = 'INTERACTIVE MIGRATION VISUALIZATION',
                                showlegend = False,
                                margin = {'b': 40, 'l': 40, 'r': 40, 't': 40},
                                xaxis = {'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis = {'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                height = 600)}
        return figure


    traceRecode = []  
    
    colors = list(Color('lightcoral').range_to(Color('darkred'), len(G.edges())))
    colors = ['rgb' + str(x.rgb) for x in colors]

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        weight = float(G.edges[edge]['TransactionAmt']) / max(edges['TransactionAmt']) * 10
        trace = go.Scatter(x = tuple([x0, x1, None]),
                           y = tuple([y0, y1, None]),
                           mode = 'lines',
                           line = {'width': weight},
                           marker = dict(color=colors[index]),
                           line_shape = 'spline',
                           opacity = 1)
        
        traceRecode.append(trace)
        index = index + 1
    
    node_trace = go.Scatter(x = [],
                            y = [],
                            hovertext = [],
                            text = [],
                            mode = 'markers+text',
                            textposition = "bottom center",
                            hoverinfo = "text",
                            marker = {'size': 50, 'color': 'LightSkyBlue'})

    # Add nodes
    index = 0
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        hovertext = "Nombre: " + str(G.nodes[node]['Nombre']) + "<br>" + "Tipo de entidad: " + str(G.nodes[node]['Tipo'])
        text = nodes[nodes["Nombre"] == G.nodes[node]['Nombre']]["CODMUN"].values[0]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['hovertext'] += tuple([hovertext])
        node_trace['text'] += tuple([text])
        index = index + 1

    traceRecode.append(node_trace)
    
    middle_hover_trace = go.Scatter(x = [],
                                    y = [],
                                    hovertext = [],
                                    mode = 'markers',
                                    hoverinfo = "text",
                                    marker = {'size': 20, 'color': 'LightSkyBlue'},
                                    opacity = 0)

    # Add edges
    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        hovertext = "From: " + str(G.edges[edge]['Source']) + "<br>" + "To: " + str(
            G.edges[edge]['Target']) + "<br>" + "Flujo de personas: " + str(
            G.edges[edge]['TransactionAmt']) + "<br>" + "Año: " + str(G.edges[edge]['Date'])
        middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
        middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
        middle_hover_trace['hovertext'] += tuple([hovertext])
        index = index + 1

    traceRecode.append(middle_hover_trace)
    
    figure = {
        "data": traceRecode,
        "layout": go.Layout(title = 'VISUALIZACIÓN INTERACTIVA PARA FLUJOS MIGRATORIOS',
                            showlegend = False,
                            hovermode = 'closest',
                            margin = {'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis = {'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis = {'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height = 600,
                            clickmode = 'event+select',
                            annotations=[
                                dict(
                                    ax = (G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 2,
                                    ay = (G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1]) / 2, 
                                    axref = 'x',
                                    ayref = 'y',
                                    x = (G.nodes[edge[1]]['pos'][0] * 3 + G.nodes[edge[0]]['pos'][0]) / 4,
                                    y = (G.nodes[edge[1]]['pos'][1] * 3 + G.nodes[edge[0]]['pos'][1]) / 4,
                                    xref = 'x',
                                    yref = 'y',
                                    showarrow = True,
                                    arrowhead = 3,
                                    arrowsize = 4,
                                    arrowwidth = 1,
                                    opacity = 1
                                ) for edge in G.edges])}
    return figure





styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    # Tittle
    html.Div([html.H1("GRAFO PARA FLUJOS MIGRATORIOS")],
             className="row",
             style={'textAlign': "center"}),
    ## Define row
    html.Div(
        className="row",
        children=[
            # left side
            html.Div(
                className="two columns",
                children=[
                    dcc.Markdown(d("""
                            **PERIODO DE VISUALIZACION**
                            Desliza para definir el rango en años
                            
                            Ejemplo: 10 equiv. 2010
                            """)),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.RangeSlider(
                                id='my-range-slider',
                                min=2010,
                                max=2019,
                                step=1,
                                value=[2010, 2019],
                                marks={
                                    2010: {'label': '10'},
                                    2011: {'label': '11'},
                                    2012: {'label': '12'},
                                    2013: {'label': '13'},
                                    2014: {'label': '14'},
                                    2015: {'label': '15'},
                                    2016: {'label': '16'},
                                    2017: {'label': '17'},
                                    2018: {'label': '18'},
                                    2019: {'label': '19'}
                                }
                            ),
                            html.Br(),
                            html.Div(id='output-container-range-slider')
                        ],
                        style={'height': '300px'}
                    ),
                        
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Markdown(d("""
                            **BUSCA UN MUNICIPIO**
                            
                            Introduce el código del municipio
                            """)),
                            dcc.Input(id="input1",
                                      type="text",
                                      placeholder="CODMUN"),
                            html.Div(id="output")
                        ],
                        style={'height': '300px'}
                    )
                ]
            ),

            
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph",
                                    figure=network_graph(PERIOD, POPULATION_CENTRE))],
            ),

            ################## followig PART CAN BE REMOVED ###################
            html.Div(
                className="two columns",
                children=[
                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Hover Data**
                            Mouse over values in the graph.
                            """)),
                            html.Pre(id='hover-data', style=styles['pre'])
                        ],
                        style={'height': '400px'}),

                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Click Data**
                            Click on points in the graph.
                            """)),
                            html.Pre(id='click-data', style=styles['pre'])
                        ],
                        style={'height': '400px'})
                ]
            )
        ]
    )
])

## CALLBACK FOR LEFT SIDE MEMBERS
@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('my-range-slider', 'value'),
     dash.dependencies.Input('input1', 'value')])

def update_output(value,input1):
    # to update the global variables of PERIOD and POPULATION_CENTRE
    PERIOD = value
    POPULATION_CENTRE = input1
    return network_graph(value, input1)

    

      
########################## THIS PART CAN BE REMOVED ##########################
@app.callback(
    dash.dependencies.Output('hover-data', 'children'),
    [dash.dependencies.Input('my-graph', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@app.callback(
    dash.dependencies.Output('click-data', 'children'),
    [dash.dependencies.Input('my-graph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)
##############################################################################



if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(#debug=True,
               host = "127.0.0.1",
               port = "8050")