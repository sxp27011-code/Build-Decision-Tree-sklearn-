**700762701 PUDARI SREE RAM CHARAN TEJA**
# Build-Decision-Tree-sklearn-
# 🌳 Decision Tree Classification on Iris Dataset (sklearn)

This project demonstrates how to build and analyze **Decision Tree Classifiers** using the Iris dataset.  
We compare models with different tree depths (`max_depth = 1, 2, 3`) and discuss signs of **underfitting** vs **overfitting**.

---

## 📌 Problem Statement

1. Use `sklearn.tree.DecisionTreeClassifier` on the Iris dataset.  
2. Train decision trees with `max_depth = 1, 2, 3`.  
3. Report **training** and **test** accuracy for each depth.  
4. Plot the learned decision tree structure.  
5. Discuss signs of **underfitting** vs **overfitting**.

---

## 💻 Code

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Decision Trees with different depths
depths = [1, 2, 3]
results = []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    results.append((d, train_acc, test_acc))
    
    # Print results
    print(f"Depth = {d}")
    print(f"  Training accuracy = {train_acc:.4f}")
    print(f"  Test accuracy     = {test_acc:.4f}")
    
    # Plot decision tree
    plt.figure(figsize=(8, 4))
    plot_tree(clf, feature_names=iris.feature_names,
              class_names=iris.target_names,
              filled=True, rounded=True, fontsize=10)
    plt.title(f"Decision Tree (max_depth={d})")
    plt.show()

# Summary
print("\nSummary:")
print("Depth | Train Acc | Test Acc")
for d, ta, te in results:
    print(f"{d:<5} | {ta:.4f}     | {te:.4f}")
