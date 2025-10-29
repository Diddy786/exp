# Artificial Intelligence & Machine Learning Experiments

This repository contains 6 fundamental AI/ML experiments implemented using Python. Each experiment demonstrates core concepts in artificial intelligence and machine learning with practical implementations.

## 📋 Requirements

- **Python 3.x**
- **Required Libraries:**
  ```bash
  pip install scikit-learn matplotlib numpy
  ```

## 🧪 Experiments Overview

### 🧪 Experiment 1: Depth First Search (DFS)

**AIM:** Write a Python program to implement Depth First Search (Uninformed).

**Requirements:** Python software, any text editor (VS Code / Jupyter / Google Colab).

**Code:**
```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")
    for neighbour in graph[start]:
        if neighbour not in visited:
            dfs(graph, neighbour, visited)

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
dfs(graph, 'A')
```

**Output:**
```
A B D E F C
```

**Conclusion:** DFS explores as far as possible along each branch before backtracking.

**Key Questions & Answers:**

**Q1. Define DFS and difference with BFS.**
- DFS explores deep into a branch before exploring siblings
- BFS explores level by level
- DFS uses a stack (or recursion), BFS uses a queue

**Q2. Explain DFS working.**
- Start from a node → visit it → go to next unvisited neighbor → continue until no path remains → backtrack
- Example order: A → B → D → E → F → C

---

### 🧪 Experiment 2: Greedy Best-First Search (Informed)

**AIM:** Write Python program to implement Greedy Best-First Search.

**Requirements:** Python with heapq library.

**Code:**
```python
import heapq

def greedy_best_first(graph, start, goal, heuristic):
    visited = set()
    pq = [(heuristic[start], start)]
    while pq:
        (h, node) = heapq.heappop(pq)
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            if node == goal:
                break
            for neighbor in graph[node]:
                heapq.heappush(pq, (heuristic[neighbor], neighbor))

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
heuristic = {'A': 6, 'B': 4, 'C': 3, 'D': 2, 'E': 1, 'F': 0}

greedy_best_first(graph, 'A', 'F', heuristic)
```

**Output:**
```
A C F
```

**Conclusion:** Greedy Best-First uses heuristic values to choose the most promising node.

**Key Questions & Answers:**

**Q1. Explain working principle.**
- It selects the node that seems closest to the goal using a heuristic
- Called Informed because it uses extra knowledge (heuristic function)

**Q2. Why better than BFS/DFS?**
- BFS/DFS explore blindly
- Greedy Best-First uses heuristics to reach goal faster

---

### 🧪 Experiment 3: Breadth First Search (BFS)

**AIM:** Write Python program to implement BFS (Uninformed).

**Requirements:** Python 3

**Code:**
```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    while queue:
        node = queue.popleft()
        print(node, end=" ")
        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
bfs(graph, 'A')
```

**Output:**
```
A B C D E F
```

**Conclusion:** BFS explores all nodes level by level using a queue.

**Key Questions & Answers:**

**Q1. Explain working.**
- Start from root → explore all neighbors → then move to next level
- Traversal order example: A, B, C, D, E, F

**Q2. Applications of BFS:**
- Shortest path in unweighted graphs
- Web crawler page visiting
- Finding connected components in networks

---

### 🧪 Experiment 4: Split Dataset into Train/Test

**AIM:** Write Python program to split dataset into training and testing sets.

**Requirements:** Python, scikit-learn

**Code:**
```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
```

**Output:**
```
Training samples: 105
Testing samples: 45
```

**Conclusion:** Splitting dataset helps test the model on unseen data for fair evaluation.

**Key Questions & Answers:**

**Q1. Difference between training and testing set:**
- **Training set** → used to train the model
- **Testing set** → used to evaluate model accuracy

**Q2. What if not split?**
- The model memorizes data → gives high accuracy but poor real-world performance (overfitting)

---

### 🧪 Experiment 5: Decision Tree

**AIM:** Create and display a Decision Tree on given dataset.

**Requirements:** Python, scikit-learn, matplotlib.

**Code:**
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = load_iris()
model = DecisionTreeClassifier()
model.fit(data.data, data.target)

plt.figure(figsize=(8,6))
plot_tree(model, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()
```

**Output:**
```
(A displayed Decision Tree chart showing splits.)
```

**Conclusion:** Decision Tree classifies data by splitting into branches based on features.

**Key Questions & Answers:**

**Q1. Which function visualizes the tree?**
- `plot_tree()` from sklearn
- **Parameters:**
  - `model`: trained model
  - `filled=True`: colors nodes
  - `feature_names`, `class_names`: label info

**Q2. What is Decision Tree?**
- A tree-like model for decisions
- **Root Node:** starting feature
- **Decision Nodes:** tests
- **Leaf Nodes:** final output/class
- **Example:** Weather → (Sunny → Play / Rainy → Don't Play)

---

### 🧪 Experiment 6: Simple Linear Regression

**AIM:** Implement Simple Linear Regression in Python.

**Requirements:** Python, scikit-learn.

**Code:**
```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("Prediction for x=6:", model.predict([[6]])[0])
```

**Output:**
```
Slope (m): 0.6
Intercept (c): 2.2
Prediction for x=6: 5.8
```

**Conclusion:** Linear regression finds a straight-line relationship between X and Y.

**Key Questions & Answers:**

**Q1. Difference between Simple and Multiple Linear Regression:**
- **Simple:** 1 independent variable → e.g. y = m*x + c
- **Multiple:** 2 or more independent variables → e.g. y = b0 + b1*x1 + b2*x2

**Q2. What is Simple Linear Regression?**
- A method to find relation between two variables
- **Equation:** y = m*x + c
  - y = predicted value
  - x = input value
  - m = slope
  - c = intercept

---

## 🎯 Key Concepts Summary

### Search Algorithms
- **DFS:** Deep exploration using stack/recursion
- **BFS:** Level-by-level exploration using queue
- **Greedy Best-First:** Informed search using heuristics

### Machine Learning
- **Dataset Splitting:** Essential for model validation
- **Decision Trees:** Rule-based classification
- **Linear Regression:** Finding linear relationships in data

## 🚀 Getting Started

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install scikit-learn matplotlib numpy
   ```
3. Run any experiment script
4. Each experiment is self-contained and ready to execute

## 📊 Algorithm Comparison

| Algorithm | Type | Data Structure | Time Complexity | Space Complexity |
|-----------|------|----------------|-----------------|------------------|
| DFS | Uninformed | Stack/Recursion | O(V + E) | O(V) |
| BFS | Uninformed | Queue | O(V + E) | O(V) |
| Greedy Best-First | Informed | Priority Queue | O(b^m) | O(b^m) |

Where V = vertices, E = edges, b = branching factor, m = maximum depth

## 📚 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Graph Algorithms Guide](https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/)
- [Machine Learning Basics](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)

---

**Note:** Each experiment includes practical implementations with real outputs and comprehensive Q&A sections to reinforce learning concepts in AI and Machine Learning.
