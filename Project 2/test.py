# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import preprocessing
from sklearn import decomposition
from train import glucoseFeatures
import pickle
import pickle_compat
pickle_compat.patch()

with open("model.pkl", 'rb') as file:
        loaded_model = pickle.load(file) 
        test_df = pd.read_csv('test.csv', header=None)
    
cgm_features=glucoseFeatures(test_df)
ss_fit = preprocessing.StandardScaler().fit_transform(cgm_features)
    
pca = decomposition.PCA(n_components=5)
pca_fit=pca.fit_transform(ss_fit)
    
results = loaded_model.predict(pca_fit)
pd.DataFrame(results).to_csv("Results.csv", header=None, index=False)
