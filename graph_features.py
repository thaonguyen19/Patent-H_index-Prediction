"""
Python file to generate features for networks created from graph_generation.

graph_generation.py should be run before running this file.
"""

from collections import namedtuple
import os
import snap
# TODO? jaccard_index, avg_path_len, betweenness
NetworkStats = namedtuple("NetworkStats", "node_count edge_count clustering_cf num_sccs max_scc_proportion")

network_folder = '../data/networks/'

class AssigneeGraph(object):
	def __init__(self, company_name, Graph):
		self.company_name = company_name
		self.Graph = Graph


# Initial step: Load all generated network files
def load_networks(folder, file_list=None):
	AssigneeGraphs = []
	if not file_list:
		file_list = []
		# Load all networks in folder
		for file in os.listdir(folder):
			if file.endswith(".txt"):
				file_list.append(os.path.join(folder, file))
	for filename in file_list:
		Graph = snap.LoadEdgeList(snap.PUNGraph, filename, 0, 1, '\t')
		AssigneeGraphs.append(AssigneeGraph(os.path.basename(filename), Graph))
	return AssigneeGraphs


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
		net_stats = NetworkStats(node_count=node_count, edge_count=edge_count, clustering_cf=cc, num_sccs=num_sccs, max_scc_proportion=max_scc_proportion)
		print AGraph.company_name
		print(net_stats)

main()