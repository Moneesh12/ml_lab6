import pandas as pd
import numpy as np

#dataset

rf = pd.read_csv("DCT_mal.csv")

target = "LABEL"

#equal width binning 
def custom_binning(series, bins=3, method="equal_width"):
    series = series.dropna() 

    #equal width or equal frequency
    
    if method == "equal_width":

        min_val, max_val = series.min(), series.max()
        bin_width = (max_val - min_val) / bins
        edges = [min_val + i * bin_width for i in range(bins+1)]
    
    elif method == "equal_frequency":

        edges = [series.quantile(i / bins) for i in range(bins+1)]
    
    else:
        raise ValueError("Method must be 'equal_width' or 'equal_frequency'")
    

    labels = [f"Bin{i+1}" for i in range(bins)]
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True)

for col in rf.columns:
    if rf[col].dtype in ["float64", "int64"] and col != target:
        rf[col] = custom_binning(rf[col], bins=3, method="equal_frequency")  # try "equal_width" too

#finding entropy
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

#calculate information gain
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
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
