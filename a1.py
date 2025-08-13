import numpy as np
import pandas as pd

rf = pd.read_csv('DCT_mal.csv')

x = rf.drop(columns='LABEL').values
y = rf['LABEL'].values

counts = rf['LABEL'].value_counts().values
probability = counts / counts.sum()
entropy = -np.sum(probability * np.log2(probability))

print("entropy value:",entropy)