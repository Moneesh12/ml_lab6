import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

#dataset
rf = pd.read_csv("DCT_mal.csv")

#x for features and y for label
X = rf.iloc[:, [0, 1]]
y = rf.iloc[:, -1]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Initialize decision tree with entropy, fixed random state, and constraints
clf = DecisionTreeClassifier(
    criterion="entropy",
    random_state=0,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)
#training the decision tree model
clf.fit(X, y)

# decision tree with feature names, class names, filled nodes, and rounded edges
plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=X.columns.tolist(),
    class_names=[str(c) for c in sorted(y.unique())],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.show()