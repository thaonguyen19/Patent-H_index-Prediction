"""
Python file to generate co-authorship networks based on data from USPTO PatentsView

You will need the following files to run this code:
    - assignee.tsv
    - inventor.tsv
    - patent.tsv
    - patent_inventor.tsv
    - patent_assignee.tsv
"""

import json
import snap
import pandas as pd
import numpy as np
import datetime
import re

#Path to folder with all the .tsvs downloaded from PatentsView
data_folder = '../data/'
out_folder = '../data/networks/'

print "Loading Data..."

df_assignee = pd.read_csv(data_folder + 'assignee.tsv', delimiter='\t', header=0, usecols=['id', 'name_first', 'name_last', 'organization'])
df_inventor = pd.read_csv(data_folder + 'inventor.tsv', delimiter='\t', header=0, dtype=str)
df_patent = pd.read_csv(data_folder + 'patent.tsv', delimiter='\t', header=0, usecols=['id', 'date'])
df_patent_inventor = pd.read_csv(data_folder + 'patent_inventor.tsv', delimiter='\t', header=0)
df_patent_assignee = pd.read_csv(data_folder + 'patent_assignee.tsv', delimiter='\t', header=0) 

print "Data Loaded."

#Rename columns
df_patent.rename(columns={'id': 'patent_id'}, inplace=True)
df_assignee.rename(columns={'id': 'assignee_id'}, inplace=True)
df_inventor.rename(columns={'id': 'inventor_id'}, inplace=True)

#Convert patent date to python datetime
df_patent['date'] = pd.to_datetime(df_patent['date'], errors='coerce', format="%Y-%m-%d")

print "Filtering Data..."

#Filter all private assignees (not companies)
df_assignee = df_assignee[pd.isnull(df_assignee.name_first) & \
                          pd.isnull(df_assignee.name_last) & \
                          pd.notnull(df_assignee.organization)]

#Filter patents not not issued from 1990 - 2000
df_patent = df_patent[df_patent['date'] > datetime.datetime(1990, 1, 1)]
df_patent = df_patent[df_patent['date'] < datetime.datetime(2000, 1, 1)]
df_patent = df_patent[pd.notnull(df_patent.date)]

print "Joining Data..."

#Merge all data into one dataframe
df_all = pd.merge(df_patent_inventor, df_patent_assignee, how='inner', on='patent_id')
df_all = pd.merge(df_all, df_assignee, how='inner', on='assignee_id')
df_all = pd.merge(df_all, df_inventor, how='inner', on='inventor_id')
df_all = pd.merge(df_all, df_patent, how='inner', on='patent_id')

#Set index to make retrieval easier later
df_inventor.set_index('inventor_id', inplace=True)

#Create dataframe to count patents
df_counts = df_all[['assignee_id', 'organization']].copy()

#Count how many patents each company generated
df_counts['company_count'] = df_all.assignee_id.map(df_all.assignee_id.value_counts())
df_counts = df_counts.drop_duplicates()
df_counts = df_counts.sort_values(by='company_count', ascending=False)

#Extract company ids who produced > 100 patents in timeframe
companies = df_counts[df_counts['company_count'] > 100]['assignee_id']

df_counts.set_index('assignee_id', inplace=True)

print "Generating Graphs..."

for c in companies:
    company_name = df_counts.loc[c, 'organization']
    #Strip special characters from company_name
    company_name = re.sub('[^0-9a-zA-Z]+', '', company_name)
    Graph = snap.PUNGraph.New()
    #Extract data for company of interest
    company_info = df_all[df_all['assignee_id'] == c][['patent_id', 'inventor_id']].sort_values(by='patent_id')
    #Get all nodes (inventors)
    nodes = company_info['inventor_id'].drop_duplicates().tolist()
    #Get all potential edges (patent info)
    patents = company_info['patent_id'].drop_duplicates().tolist()
    #Hash of inventor_id -> node_id (for easier graph manipulation)
    inventor_id_to_index = snap.TStrIntH()

    metadata = {}
    metadata['number_of_patents'] = len(patents)
    #Add all nodes to graph
    for i in range(0, len(nodes)):
        Graph.AddNode(i)
        inventor_id_to_index[nodes[i]] = i
    #Add all edges to graph
    for p in patents:
        #Get small connected component formed by a patent with multiple authors
        mini_cc = company_info[company_info['patent_id'] == p]['inventor_id'].tolist()
        #Only for patents with > 1 author
        if len(mini_cc) > 1:
            #Add all possible edges for co-authorship
            for index1 in range(0, len(mini_cc)):
                for index2 in range(index1 + 1, len(mini_cc)):
                    inventor1 = inventor_id_to_index[mini_cc[index1]]
                    inventor2 = inventor_id_to_index[mini_cc[index2]]
                    #Don't add multiple edges (maybe alter later for weights)
                    if not Graph.IsEdge(inventor1, inventor2):
                        Graph.AddEdge(inventor1, inventor2)
    print snap.PrintInfo(Graph)
    with open(out_folder + '%s.json' %company_name, 'w') as fp:
        json.dump(metadata, fp, sort_keys=True, indent=4)
    #Save inventor metadata as .tsv
    df_inventor_metadata = {'inventor_id':[], 'node_id':[], 'inventor_name':[]}
    for inventor in inventor_id_to_index:
        df_inventor_metadata['inventor_id'].append(inventor)
        df_inventor_metadata['node_id'].append(inventor_id_to_index[inventor])
        inventor_name = '%s %s' %(df_inventor.loc[inventor, 'name_first'], \
                                  df_inventor.loc[inventor,'name_last'])
        df_inventor_metadata['inventor_name'].append(inventor_name)
    df_inventor_metadata = pd.DataFrame(df_inventor_metadata)
    df_inventor_metadata.set_index('node_id')
    df_inventor_metadata.to_csv(out_folder + '%s_inventor_metadata.tsv' %company_name, sep='\t', index=False)
    snap.SaveEdgeList(Graph, out_folder + '%s.txt' %company_name, \
                      "Collaboration network for company, drawn from patent data")
    print "Saved data for %s" %company_name
