"""
Python file to generate features for networks created from graph_generation.

graph_generation.py should be run before running this file.
"""

from collections import namedtuple
import cPickle as pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import snap
from tqdm import tqdm

# Look into H-index from TA feedback
NetworkStats = namedtuple("NetworkStats", "node_count edge_count clustering_cf num_sccs "
    "max_scc_proportion avg_patents_per_inventor modularity")

network_folder = '../data/networks/'


class AssigneeGraph(object):
    def __init__(self, company_name, Graph, metadata):
        self.company_name = company_name
        self.Graph = Graph
        self.metadata = metadata


# Initial step: Load all generated network files
def load_networks(folder, graph_list=None):
    AssigneeGraphs = []
    if not graph_list:
        graph_list = []
        # Load all networks in folder
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                graph_list.append(os.path.join(folder, file))
    for filename in graph_list:
        Graph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1, '\t')
        meta_file = os.path.splitext(filename)[0]+'.json'
        with open(meta_file, 'r') as fp:
            metadata = json.load(fp)
        AssigneeGraphs.append(AssigneeGraph(os.path.splitext(os.path.split(filename)[1])[0], Graph, metadata))
    return AssigneeGraphs


def get_modularity(Graph):
	# Uses the Girvan-Newman community detection algorithm based on betweenness centrality on Graph.
	CmtyV = snap.TCnComV()
	modularity = snap.CommunityGirvanNewman(Graph, CmtyV)
	return modularity


def plot_hist(data, title, xlabel):
	plt.hist(data)
	plt.xlabel(xlabel)
	plt.ylabel("Frequency")
	plt.title(title)
	plt.show()

def analyze_stats(stats):
	print "Avg node count", np.mean([net_stats.node_count for net_stats in stats])
	print "Avg edge count", np.mean([net_stats.edge_count for net_stats in stats])
	# Histograms
	plot_hist([net_stats.modularity for net_stats in stats], "Modularity of Organization Patent Networks", "Modularity")
	plot_hist([net_stats.max_scc_proportion for net_stats in stats], "SCC Proportion of Organization Patent Networks", "Max SCC Proportion")
	plot_hist([net_stats.avg_patents_per_inventor for net_stats in stats], "Patents per Inventor in Organization Patent Networks", "Average Patents per Inventor")
	plot_hist([net_stats.clustering_cf for net_stats in stats], "Clustering Coeffecient of Organization Patent Networks", "Clustering Coeffecient (Watts and Strogatz)")

def save_net_stats(stats):
	with open(network_folder + "net_stats.json", 'wb') as fp:
		pickle.dump(stats, fp)

def load_net_stats():
	with open(network_folder + "net_stats.json", 'rb') as fp:
		stats = pickle.load(fp)
	return stats

def calc_net_stats():
	print "Loading networks..."
	AssigneeGraphs = load_networks(network_folder)
	stats = []
	print "Calculating features..."
	for AGraph in tqdm(AssigneeGraphs):
		# Calculate network features
		Graph = AGraph.Graph
		node_count = Graph.GetNodes()
		if node_count <= 0:
			# print "0 nodes", AGraph.company_name
			continue
		edge_count = Graph.GetEdges()
		cc = snap.GetClustCf(Graph)
		Components = snap.TCnComV()
		snap.GetSccs(Graph, Components)
		num_sccs = len(Components)
		MxScc = snap.GetMxScc(Graph)
		max_scc_proportion = float(MxScc.GetNodes()) / node_count
		avg_patents_per_inventor =float(AGraph.metadata['number_of_patents']) / node_count
		modularity = get_modularity(Graph)
		net_stats = NetworkStats(node_count=node_count, edge_count=edge_count, clustering_cf=cc,
			num_sccs=num_sccs, max_scc_proportion=max_scc_proportion,
			avg_patents_per_inventor=avg_patents_per_inventor, modularity=modularity)
		stats.append(net_stats)
	return stats

def main_net_stats():
	load = True
	if load:
		stats = load_net_stats()
	else:
		stats = calc_net_stats()
		save_net_stats(stats)
	print "Analyzing statistics"
	analyze_stats(stats)


def main():
    AssigneeGraphs = load_networks(network_folder)
    for AGraph in AssigneeGraphs:
        # Calculate network features
        Graph = AGraph.Graph
        node_count = Graph.GetNodes()
        if node_count <= 0:
            print "0 nodes", AGraph.company_name
            continue
        edge_count = Graph.GetEdges()
        cc = snap.GetClustCf(Graph)
        Components = snap.TCnComV()
        snap.GetSccs(Graph, Components)
        num_sccs = len(Components)
        MxScc = snap.GetMxScc(Graph)
        max_scc_proportion = float(MxScc.GetNodes()) / node_count
        avg_patents_per_inventor =float(AGraph.metadata['number_of_patents']) / node_count
        modularity = get_modularity(Graph)
        net_stats = NetworkStats(node_count=node_count, edge_count=edge_count, clustering_cf=cc,
            num_sccs=num_sccs, max_scc_proportion=max_scc_proportion,
            avg_patents_per_inventor=avg_patents_per_inventor, modularity=modularity)
        AGraph.metadata['node_count'] = node_count
        AGraph.metadata['edge_count'] = edge_count
        AGraph.metadata['clustering_cf'] = cc
        AGraph.metadata['num_sccs'] = num_sccs
        AGraph.metadata['max_scc_proportion'] = max_scc_proportion
        AGraph.metadata['avg_patents_per_inventor'] = avg_patents_per_inventor
        AGraph.metadata['modularity'] = modularity
        with open(network_folder + '%s.json' %AGraph.company_name, 'w') as fp:
            json.dump(AGraph.metadata, fp, sort_keys=True, indent=4)
    print len(AssigneeGraphs)

main()