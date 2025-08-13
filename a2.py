import numpy as np
import pandas as pd
import math as m
rf = pd.read_csv('DCT_mal.csv')

x = rf.drop(columns='LABEL').values
y = rf['LABEL'].values

counts = rf['LABEL'].value_counts().values
probability = counts / counts.sum()
entropy = np.sum(probability * probability)

gini = m.pow(entropy,2)

gini1 = 1 - entropy

print("the gini value:",gini1)