import numpy as np
import pandas as pd

#dataset
rf = pd.read_csv('DCT_mal.csv')

#split the dataset
x = rf.drop(columns='LABEL').values
y = rf['LABEL'].values

#entropy formula
counts = rf['LABEL'].value_counts().values
probability = counts / counts.sum()
entropy = -np.sum(probability * np.log2(probability))

print("entropy value:",entropy)
