'''
Python file to generate visualization for networks - change the variables at the
beginning for in folder for network .txt files and out folder for .png
'''

import snap
import os
import json
import re

IN_FOLDER = '../data/networks'
OUT_FOLDER = '../data/viz/'

def load_networks(folder):
    graph_list = []
    snap_graphs = []
    # Load all networks from folder
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            graph_list.append(os.path.join(folder, file))
    for filename in graph_list:
        Graph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1, '\t')
        snap_graphs.append(Graph)
    graph_list = [re.search(r'(/)([a-zA-Z]*)(\.txt)', n).group(2) \
                                                            for n in graph_list]
    return snap_graphs, graph_list

def main():
    graphs, names = load_networks(IN_FOLDER)
    for g,n in zip(graphs, names):
        NIdColorH = snap.TIntStrH()
        for i in g.Nodes():
            NIdColorH[i.GetId()] = "lightblue"
        snap.DrawGViz(g, snap.gvlCirco, OUT_FOLDER + n + ".png", n, False, NIdColorH)

main()