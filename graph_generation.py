import snap
import pandas as pd
import numpy as np

data_folder = 'data/' 

print "Loading Data..."

df_assignee = pd.read_csv(data_folder + 'assignee.tsv', delimiter='\t', header=0)
df_inventor = pd.read_csv(data_folder + 'inventor.tsv', delimiter='\t', header=0)
df_patent_inventor = pd.read_csv(data_folder + 'patent_inventor.tsv', delimiter='\t', header=0)
df_patent_assignee = pd.read_csv(data_folder + 'patent_assignee.tsv', delimiter='\t', header=0)

print "Data Loaded."

print "Filtering Data..."

df_assignee = df_assignee[pd.isnull(df_assignee.name_first) & \
                          pd.isnull(df_assignee.name_last) & \
                          pd.notnull(df_assignee.organization)]

df_assignee = df_assignee.drop(['name_first','name_last', 'type'], axis=1)

print "Joining Data..."

df_all = pd.merge(df_patent_inventor, df_patent_assignee, how='inner', on='patent_id')
df_all = pd.merge(df_all, df_assignee, how='inner', on='assignee_id')
df_all = pd.merge(df_all, df_inventor, how='inner', on='inventor_id')

df_counts = df_all[['assignee_id', 'organization']].copy()

df_counts['company_count'] = df_all.assignee_id.map(df_all.assignee_id.value_counts())
df_counts = df_counts.drop_duplicates()
df_counts = df_counts.sort_values(by='company_count', ascending=False)
df_counts.set_index('assignee_id')
companies = df_counts.head(5)['assignee_id'].tolist()

print "Generating Graphs..."

for c in companies:
    Graph = snap.PUNGraph.New()
    company_info = df_all[df_all['assignee_id'] == c][['patent_id', 'inventor_id']].sort_values(by='patent_id')
    nodes = company_info['inventor_id'].drop_duplicates().tolist()
    patents = company_info['patent_id'].drop_duplicates().tolist()
    inventor_id_to_index = snap.TStrIntH()
    for i in range(0, len(nodes)):
        Graph.AddNode(i)
        inventor_id_to_index[nodes[i]] = i
    for p in patents:
        mini_cc = company_info[company_info['patent_id'] == p]['inventor_id'].tolist()
        if len(mini_cc) > 1:
            for index1 in range(0, len(mini_cc)):
                for index2 in range(index1 + 1, len(mini_cc)):
                    inventor1 = inventor_id_to_index[mini_cc[index1]]
                    inventor2 = inventor_id_to_index[mini_cc[index2]]
                    if not Graph.IsEdge(inventor1, inventor2):
                        Graph.AddEdge(inventor1, inventor2)
    print snap.PrintInfo(Graph)
    snap.SaveEdgeList(Graph, data_folder + '%s.txt' %df_assignee[df_assignee['assignee_id'] == c]['organization'], \
                      "Collaboration network for company, drawn from patent data")
