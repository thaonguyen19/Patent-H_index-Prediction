"""
Python file to generate features for networks created from graph_generation.

graph_generation.py should be run before running this file.
"""

from collections import namedtuple
import json
import os
import snap


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
		AssigneeGraphs.append(AssigneeGraph(os.path.basename(filename), Graph, metadata))
	return AssigneeGraphs


def get_modularity(Graph):
	# Uses the Girvan-Newman community detection algorithm based on betweenness centrality on Graph.
	CmtyV = snap.TCnComV()
	modularity = snap.CommunityGirvanNewman(Graph, CmtyV)
	return modularity


def h_index():
	pass


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
		print AGraph.company_name
		print(net_stats)


main()