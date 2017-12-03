"""
Python file to run cross-validated linear regression prediction

You should have run graph_generation.py, graph_features.py,
and the citation feature generation before running this file.
"""

from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import os
import json
import numpy as np

network_folder = '../data/networks/'
num_features = 9

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
        company_y = data.pop('forward_citation_count')
        company_x = []
        for k in sorted(data):
            company_x.append(data[k])
        if len(company_x) == num_features:
            y.append(company_y)
            x.extend(company_x)
        else:
            rem.append(name)
    for c in rem:
        company_names.remove(c)
    x = np.array(x)
    x = x.reshape((len(company_names), len(x)/len(company_names)))
    y = np.array(y)
    company_names = [os.path.splitext(os.path.split(n)[1])[0] for n in company_names]
    return company_names, x, y

def main():
    company_names, X, Y = load_data(network_folder)
    lr = linear_model.LinearRegression()
    Y_pred = cross_val_predict(lr, X, Y, cv=10)
    Y_mean = np.mean(Y)
    Y_pred_mean = np.mean(Y_pred)
    nope=0
    for pred_pair in zip(Y, Y_pred):
        print "Actual: %s, Predicted: %s" %pred_pair
        if pred_pair[0] > Y_mean != pred_pair[1] > Y_pred_mean:
            nope+= 1
    print "%s / %s" %(nope, len(Y))
    print mean_squared_error(Y, Y_pred)

main()