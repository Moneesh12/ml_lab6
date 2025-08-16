import pandas as pd
import numpy as np

#dataset

rf = pd.read_csv("DCT_mal.csv")

target = "LABEL"


#features into categorical bins
for col in rf.columns:
    if rf[col].dtype in ["float64", "int64"] and col != target:
        rf[col] = pd.qcut(rf[col], q=3, labels=["Low", "Medium", "High"])

# entropy function
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

#calculating information gain
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])

# find weighted entropy for each value of the feature
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset = data[data[feature] == v]
        weighted_entropy += (c / len(data)) * entropy(subset[target])

    return total_entropy - weighted_entropy

# Calculate information gain for each feature
info_gains = {}
for feature in rf.columns:
    if feature != target:
        ig = information_gain(rf, feature, target)
        info_gains[feature] = ig

best_feature = max(info_gains, key=info_gains.get)

print("Information Gain for each feature:")
for f, ig in info_gains.items():
    print(f"{f}: {ig:.4f}")

print("\nBest feature for root node:", best_feature)
