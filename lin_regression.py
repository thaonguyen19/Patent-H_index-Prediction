"""
Python file to run cross-validated linear regression prediction

You should have run graph_generation.py, graph_features.py,
and the citation feature generation before running this file.

Inputs are json files named by company
"""

from sklearn.model_selection import cross_val_predict
from sklearn import linear_model, ensemble, feature_selection
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.feature_selection import f_regression
import os
import json
import numpy as np
import random
from scipy.stats import spearmanr
#metric to compare Y and Y_pred
from sklearn.metrics import r2_score, explained_variance_score, mutual_info_score
import matplotlib.pyplot as plt

network_folder = './json/'#'../data/networks/'
#convert prediction to a ranking
use_ranking = False
#how many features to reduce to (set = to num_components for no pca)
use_pca = False
pca_components = 6

remove_features = ['edge_count', 'node_count']
num_features = 9 - len(remove_features)

def load_data(folder):
    x = []
    y = []
    y_citation = []
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
        citation = data.pop('forward_citation_count')
        for feature in remove_features:
            if feature in data:
                data.pop(feature)

        company_x = []
        for k in sorted(data):
            company_x.append(data[k])
        #Filter out the companies without full complement of features
        if len(company_x) == num_features:
            y_citation.append(citation)
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
    return company_names, x, y, y_citation

def convert_to_rank(Y):
    Y_sorted = sorted(Y)
    Y_idx = [Y_sorted.index(y) for y in Y]
    return Y_idx

def outliers_z_score(ys, threshold=3):
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

def main():
    company_names, X, Y, Y_citation = load_data(network_folder)
    # #X = normalize(X, axis=1)

    lr = linear_model.HuberRegressor(epsilon=1.5)
    if use_pca:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
    #Run k-fold cross validation and prediction simultaneously
    Y_pred = cross_val_predict(lr, X, Y, cv=8)
    if use_ranking:
        Y = convert_to_rank(Y)
        Y_pred = convert_to_rank(Y_pred)
    #F_scores, p_values = f_regression(X, Y)
    #print F_scores
    #print p_values
    #for pred_pair in zip(Y, Y_pred):
    #    print "Actual: %s, Predicted: %s" %pred_pair
    print "Mean Absolute Error: %s" %mean_absolute_error(Y, Y_pred)
    print "Ground Truth StdDev: %s" %np.std(Y)
    lr.fit(X, Y_pred)
    print lr.coef_
    # ### Normal test stuffs
    #from scipy.stats.mstats import normaltest
    #print normaltest(Y - Y_pred)
    #plt.hist(Y - Y_pred, bins=10, color='blue')
    #plt.show()

    ### Feature multicollinearity/ redundancy test stuffs
    # corr = np.corrcoef(X, rowvar=False)
    # print corr
    # W,V=np.linalg.eig(corr)
    # print W

    # ### Plot stuffs
    # plt.hist(Y_citation, bins=20, color='blue')
    # plt.show()
    # plt.hist(Y, bins=20, color='red')
    # plt.show()

    # ### Outlier stuffs
    # outlier_ind = outliers_iqr(Y)
    # print "NUMBER OF OUTLIERS BY IQR: ", len(outlier_ind)
    # print "NUMBER OF OUTLIERS FOUND BY LR: ", sum(np.array(lr.outliers_))
    # for ind in outlier_ind:
    #    print lr.outliers_[ind]

    # ### Rank Correlation stuffs
    # print spearmanr(Y, Y_pred)
    
main()