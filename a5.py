import pandas as pd
import numpy as np

#initialising nodes 
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

#finding entropy
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -(p * np.log2(p)).sum()

# Calculate information gain for a given feature
def info_gain(x, y, thresh):
    left = y[x <= thresh]
    right = y[x > thresh]
    if len(left) == 0 or len(right) == 0:
        return 0
    h = entropy(y)
    hl = entropy(left)
    hr = entropy(right)
    return h - (len(left)/len(y))*hl - (len(right)/len(y))*hr

# Finding the best feature and threshold for splitting
def best_split(X, y, bins=10):
    best_feat, best_thresh, best_gain = None, None, -1
    n, m = X.shape
    for feat in range(m):
        vals = np.linspace(X[:, feat].min(), X[:, feat].max(), bins)
        for t in vals:
            g = info_gain(X[:, feat], y, t)
            if g > best_gain:
                best_gain = g
                best_feat = feat
                best_thresh = t
    return best_feat, best_thresh, best_gain

#building the tree
def build_tree(X, y, depth=0, max_depth=3):
    if len(set(y)) == 1 or depth == max_depth:
        vals, counts = np.unique(y, return_counts=True)
        return Node(value=vals[np.argmax(counts)])
    feat, thresh, gain = best_split(X, y)
    if gain == 0:
        vals, counts = np.unique(y, return_counts=True)
        return Node(value=vals[np.argmax(counts)])
    left_idx = X[:, feat] <= thresh
    right_idx = ~left_idx
    left = build_tree(X[left_idx], y[left_idx], depth+1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth+1, max_depth)
    return Node(feature=feat, threshold=thresh, left=left, right=right)

#predicting class for single instance and then later for all
def predict_one(node, x):
    while node.value is None:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])

data = pd.read_csv("DCT_mal.csv")
X = data.drop("LABEL", axis=1).values
y = data["LABEL"].values

tree = build_tree(X, y, max_depth=3)
preds = predict(tree, X)
acc = (preds == y).mean()
print("Accuracy:", acc)
