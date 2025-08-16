import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

#dataset
df = pd.read_csv("DCT_mal.csv")

#x for features and y for label
X = df.drop(columns=["LABEL"]) 
y = df["LABEL"]

# Initialize and train decision tree classifier with entropy criterion
model = DecisionTreeClassifier(criterion="entropy", max_depth=5)
model.fit(X, y)

plt.figure(figsize=(15,8))
plot_tree(model, feature_names=X.columns, class_names=[str(c) for c in y.unique()], filled=True)
plt.show()
