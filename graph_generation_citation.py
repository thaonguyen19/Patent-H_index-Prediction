'''
Generate citation networks
'''
import json
import snap
import csv
import collections
import datetime
from tqdm import tqdm, trange
from pathlib2 import Path

data_folder = "../data/"
out_folder = '../data/networks/'

print "Loading data..."

citation_cache = collections.defaultdict(list) # map each patent to it's citations
  
# uspatentcitation_filtered_cols.tsv is generated from uspatentcitation.tsv with filter_uspatentcitation.bash
with open(data_folder + '/uspatentcitation_filtered_cols.tsv') as tsvfile:
  reader = csv.reader(tsvfile, delimiter='\t')
  next(reader, None)  # skip the headers
  for row in tqdm(reader):
    citation_cache[row[0]].append(row[1])

df_filtered = pd.read_table(data_folder + '/patent_organization_filtered.tsv', usecols=['patent_id', 'organization'])

org_patent_cache = collections.defaultdict(list) # map each firm to the patents that it owns. patent already filtered to be in 1990-2000
patentid_col = df_filtered['patent_id']
organization_col = df_filtered['organization']

for i in trange(len(patentid_col)):
  org = organization_col[i]
  patent = patentid_col[i]
  org_patent_cache[org].append(patent)

# Create backward citation graphs for each organization
for company_name, patents in org_patent_cache.iteritems():
  Graph = snap.PUNGraph.New()

  # Merge list of patents from this company and external patents they cite
  citation_map = {}
  patent_set = set()
  for patent in patents:
    patent_set.add(patent) # This patent
    patent_set.update(citation_cache[patent]) # This patent's citations
    citation_map[patent] = citation_cache[patent]

  patent_nid_map = {}
  # Add all nodes to graph
  for i, patent in enumerate(patent_set):
    patent_nid_map[patent] = i
    Graph.AddNode(i)

  # Add all backward citation edges
  for patent, citations in citation_map.iteritems():
    for cite in citations:
      Graph.AddEdge(patent_nid_map[patent], patent_nid_map[cite])

  snap.PrintInfo(Graph)
  snap.SaveEdgeList(Graph, out_folder + '{}_citation.txt'.format(company_name), \
                      "Backward citation network for company, drawn from patent data")
  print "Saved data for {}".format(company_name)
