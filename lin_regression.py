"""
Python file to run cross-validated linear regression prediction

You should have run graph_generation.py, graph_features.py,
and the citation feature generation before running this file.

Inputs are json files named by company
"""

from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
import os
import json
import numpy as np

network_folder = '../data/citation_networks/'
num_features = 8
#convert prediction to a ranking
use_ranking = False
#how many features to reduce to (set = to num_components for no pca)
pca_components = 8

def load_data(folder):
    x = []
    y = []
    company_names = []
    rem = []
    # Load all networks in folder
    for file in os.listdir(folder):
        if file.endswith(".json"):
            company_names.append(os.path.join(folder, file))
    for name in company_names:
        #Load each company's JSON file
        with open(name, 'r') as fp:
            data = json.load(fp)
        company_y = data.pop('hindex')
        data.pop('forward_citation_count') 
        company_x = []
        for k in sorted(data):
            company_x.append(data[k])
        #Filter out the companies without full complement of features
        print len(company_x)
        if len(company_x) == num_features:
            y.append(company_y)
            x.extend(company_x)
        else:
            rem.append(name)
    for c in rem:
        company_names.remove(c)
    #Convert to ndarrays
    x = np.array(x)
    x = x.reshape((len(company_names), len(x)/len(company_names)))
    y = np.array(y)
    company_names = [os.path.splitext(os.path.split(n)[1])[0] for n in company_names]
    return company_names, x, y

def convert_to_rank(Y):
    Y_sorted = sorted(Y)
    Y_idx = [Y_sorted.index(y) for y in Y]
    return Y_idx

def main():
    company_names, X, Y = load_data(network_folder)
    lr = linear_model.LinearRegression()
    pca = PCA(n_components=pca_components)
    X = pca.fit_transform(X)
    #Run k-fold cross validation and prediction simultaneously
    Y_pred = cross_val_predict(lr, X, Y, cv=8)
    if use_ranking:
        Y = convert_to_rank(Y)
        Y_pred = convert_to_rank(Y_pred)
    for pred_pair in zip(Y, Y_pred):
        print "Actual: %s, Predicted: %s" %pred_pair
    print "Mean Absolute Error: %s" %mean_absolute_error(Y, Y_pred)
    print "Ground Truth StdDev: %s" %np.std(Y)

main()