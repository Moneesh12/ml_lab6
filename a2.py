import numpy as np
import pandas as pd
import math as m

#dataset
rf = pd.read_csv('DCT_mal.csv')

x = rf.drop(columns='LABEL').values
y = rf['LABEL'].values

#finding entroy
counts = rf['LABEL'].value_counts().values
probability = counts / counts.sum()
entropy = np.sum(probability * probability)

#using entropy to find gini 
gini = m.pow(entropy,2)

gini1 = 1 - entropy

print("the gini value:",gini1)
