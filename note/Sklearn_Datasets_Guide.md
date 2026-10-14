# Scikit-Learn Built-In Datasets Guide

## ✅ YES! Scikit-learn has its own datasets!

Scikit-learn includes several built-in datasets perfect for learning machine learning without downloading external files.

---

## 📊 LOCAL DATASETS (No Internet Required)

### 1. **Iris Dataset** 🌸

```python
from sklearn import datasets
iris = datasets.load_iris()
```

- **Type**: Classification
- **Samples**: 150
- **Features**: 4 (sepal/petal length and width)
- **Classes**: 3 iris species (Setosa, Versicolor, Virginica)
- **Use Case**: Perfect for beginners, classic starter dataset
- **Access**: `iris.data`, `iris.target`, `iris.feature_names`, `iris.target_names`

---

### 2. **Digits Dataset** 🔢

```python
from sklearn import datasets
digits = datasets.load_digits()
```

- **Type**: Classification (Image Recognition)
- **Samples**: 1,797
- **Features**: 64 (8×8 pixel images)
- **Classes**: 10 (digits 0-9)
- **Use Case**: Handwritten digit recognition, computer vision basics
- **Access**: `digits.data`, `digits.target`, `digits.images`

---

### 3. **Wine Dataset** 🍷

```python
from sklearn import datasets
wine = datasets.load_wine()
```

- **Type**: Classification
- **Samples**: 178
- **Features**: 13 (alcohol, acidity, phenols, color intensity, etc.)
- **Classes**: 3 wine cultivars
- **Use Case**: Multi-class classification
- **Access**: `wine.data`, `wine.target`, `wine.feature_names`

---

### 4. **Breast Cancer Dataset** 🏥

```python
from sklearn import datasets
cancer = datasets.load_breast_cancer()
```

- **Type**: Classification (Binary)
- **Samples**: 569
- **Features**: 30 (tumor characteristics)
- **Classes**: 2 (malignant, benign)
- **Use Case**: Medical diagnosis, binary classification
- **Access**: `cancer.data`, `cancer.target`, `cancer.feature_names`

---

## 🎲 SYNTHETIC DATASETS (Generate on-the-fly)

### 5. **make_classification()**

```python
from sklearn import datasets
X, y = datasets.make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42
)
```

- **Type**: Generated classification data
- **Customizable**: Samples, features, classes, noise
- **Use Case**: Testing algorithms without real data

---

### 6. **make_regression()**

```python
from sklearn import datasets
X, y = datasets.make_regression(
    n_samples=100,
    n_features=10,
    n_informative=5,
    random_state=42
)
```

- **Type**: Generated regression data
- **Customizable**: Samples, features, noise
- **Use Case**: Testing regression algorithms

---

### 7. **make_circles()**

```python
from sklearn import datasets
X, y = datasets.make_circles(
    n_samples=100,
    noise=0.1,
    factor=0.5,
    random_state=42
)
```

- **Type**: Non-linear circular clusters
- **Use Case**: Testing non-linear classifiers
- **Features**: 2D by default

---

### 8. **make_moons()**

```python
from sklearn import datasets
X, y = datasets.make_moons(
    n_samples=100,
    noise=0.1,
    random_state=42
)
```

- **Type**: Crescent-shaped clusters
- **Use Case**: Testing clustering algorithms
- **Features**: 2D by default

---

### 9. **make_blobs()**

```python
from sklearn import datasets
X, y = datasets.make_blobs(
    n_samples=300,
    n_features=2,
    centers=4,
    random_state=42
)
```

- **Type**: Clustered data
- **Customizable**: Samples, features, centers
- **Use Case**: Testing K-means and clustering algorithms

---

## 🌐 FETCH DATASETS (Require Internet)

These datasets need to download from the internet on first use:

| Function                       | Description        | Samples | Features |
| ------------------------------ | ------------------ | ------- | -------- |
| `fetch_california_housing()` | House prices       | 20,640  | 8        |
| `fetch_20newsgroups()`       | News articles      | ~20,000 | Text     |
| `fetch_olivetti_faces()`     | Face images        | 400     | 4,096    |
| `fetch_covtype()`            | Forest cover types | 581,012 | 54       |
| `fetch_kddcup99()`           | Network intrusion  | 494,021 | 41       |
| `fetch_lfw_people()`         | Labeled faces      | 1,288   | 1,850    |
| `fetch_lfw_pairs()`          | Face pairs         | 600     | -        |
| `fetch_rcv1()`               | Reuters news       | 804,414 | -        |

---

## 📝 COMPLETE CODE EXAMPLES

### Example 1: Basic Classification with Iris

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")  # Output: ~0.97

# Confusion matrix
print(confusion_matrix(y_test, y_pred))
```

---

### Example 2: Image Classification with Digits

```python
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Cross-validation
scores = cross_val_score(pipeline, X, y, cv=5)
print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
# Output: Accuracy: 0.97 (+/- 0.01)
```

---

### Example 3: Clustering with Synthetic Data

```python
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate data
X, y_true = datasets.make_blobs(
    n_samples=300,
    centers=4,
    random_state=42
)

# Train K-means
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

# Evaluate
silhouette = silhouette_score(X, y_pred)
print(f"Silhouette Score: {silhouette:.2f}")
```

---

### Example 4: Regression with Synthetic Data

```python
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate data
X, y = datasets.make_regression(
    n_samples=100,
    n_features=10,
    noise=10,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Evaluate
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
```

---

### Example 5: Medical Diagnosis with Breast Cancer Dataset

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Load dataset
cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test),
                          target_names=cancer.target_names))
```

---

## 🎯 QUICK REFERENCE: Which Dataset to Use?

| Goal                           | Recommended Dataset                  |
| ------------------------------ | ------------------------------------ |
| Getting started with ML        | **Iris** (simplest)            |
| Learning classification        | Iris, Wine, or Breast Cancer         |
| Learning clustering            | Digits or make_blobs()               |
| Learning regression            | make_regression()                    |
| Testing non-linear classifiers | make_circles() or make_moons()       |
| Image/Computer vision basics   | Digits                               |
| Multi-class classification     | Wine or Digits                       |
| Binary classification          | Breast Cancer                        |
| Medical data prediction        | Breast Cancer                        |
| Feature engineering practice   | Wine (13 features)                   |
| Advanced ML                    | Breast Cancer (30 features)          |
| Testing with large data        | make_classification(n_samples=10000) |

---

## 🔍 How to Explore a Dataset

```python
from sklearn import datasets
import pandas as pd

# Load dataset
iris = datasets.load_iris()

# Convert to DataFrame for easy exploration
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].map({
    0: iris.target_names[0],
    1: iris.target_names[1],
    2: iris.target_names[2]
})

# Explore
print(df.head())
print(df.info())
print(df.describe())
print(df['target_name'].value_counts())
```

---

## 💡 Pro Tips

1. **Always import correctly:**

   ```python
   from sklearn import datasets  # For load_* functions
   # or
   from sklearn.datasets import load_iris, make_classification
   ```
2. **Set random_state for reproducibility:**

   ```python
   X, y = datasets.make_classification(random_state=42)
   ```
3. **For beginners, start with Iris:**

   - Small dataset (150 samples)
   - Simple features (4 measurements)
   - Clear classes (3 flower types)
   - Perfect for understanding ML workflow
4. **Use make_* functions to test hyperparameters:**

   ```python
   # Generate data with specific properties
   X, y = datasets.make_classification(
       n_samples=1000,
       n_features=50,
       n_informative=30,  # Only 30 are useful
       random_state=42
   )
   ```
5. **Combine with preprocessing:**

   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

---

## 📚 Further Learning

- **Official Scikit-Learn Documentation**: https://scikit-learn.org/stable/modules/datasets.html
- **Dataset Loading Utilities**: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.html
- **Tutorial**: https://scikit-learn.org/stable/auto_examples/index.html

---

## Summary

✅ **Scikit-learn has built-in datasets!**

- **9 local datasets** available without internet
- **8+ fetch datasets** available with internet
- **Synthetic generators** for unlimited custom data
- Perfect for learning, testing, and practicing ML algorithms

Start with **Iris** if you're a beginner, then explore others based on your learning goals!
