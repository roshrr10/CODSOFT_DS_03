#!/usr/bin/env python
# coding: utf-8

# # TASK 3

# AIM: To train a Random Forest classifier to predict the species of Iris flowers based on their measurements (sepal length, sepal width, petal length, and petal width).
# Evaluate the performance of the trained model.
# Visualize various aspects of the dataset to gain insights into the relationships between features and the distribution of data points.

# In[1]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[2]:


# Load the dataset
file_path = "C:/Users/rosha/Downloads/IRIS.csv"
iris_df = pd.read_csv(file_path)


# In[3]:


# Display the first few rows of the dataset
print(iris_df.head())


# In[4]:


# Data preprocessing
X = iris_df.drop('species', axis=1)  # Features
y = iris_df['species']  # Target variable


# In[5]:


# Convert the categorical target variable into numerical using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[6]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# In[8]:


# Predictions on the testing set
y_pred = clf.predict(X_test)


# In[9]:


# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# The Random Forest classifier achieved a high accuracy score on the testing set, indicating good predictive performance.

# In[10]:


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[11]:


# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


# In[12]:


# Feature importance
feature_importance = clf.feature_importances_
print("Feature Importance:")
for i, feature in enumerate(X.columns):
    print(feature, ':', feature_importance[i])


# In[13]:


# Visualizing Feature Importance
plt.figure(figsize=(10, 6))
plt.bar(X.columns, feature_importance)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()


# The feature importance analysis revealed that petal length and petal width are the most important features for classification.

# In[14]:


# Visualizing Decision Boundary (modified)
def plot_decision_boundary(clf, X, y, feature1, feature2):
    plt.figure(figsize=(8, 6))
    h = .02  # step size in the mesh
    x_min, x_max = X[feature1].min() - 1, X[feature1].max() + 1
    y_min, y_max = X[feature2].min() - 1, X[feature2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[feature1], X[feature2], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title('Decision Boundary')
    plt.show()


# In[15]:


import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names")

# Create DataFrames with explicit column names
X_train = pd.DataFrame(X_train, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
X_test = pd.DataFrame(X_test, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# Separate classifiers for sepal and petal measurements
clf_sepal = RandomForestClassifier(n_estimators=100, random_state=42)
clf_sepal.fit(X_train[['sepal_length', 'sepal_width']], y_train)

clf_petal = RandomForestClassifier(n_estimators=100, random_state=42)
clf_petal.fit(X_train[['petal_length', 'petal_width']], y_train)

# Visualize decision boundary for sepal measurements
plot_decision_boundary(clf_sepal, X_train, y_train, 'sepal_length', 'sepal_width')

# Visualize decision boundary for petal measurements
plot_decision_boundary(clf_petal, X_train, y_train, 'petal_length', 'petal_width')


# Decision boundary visualizations show clear separation between different species based on sepal and petal measurements.

# In[16]:


import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn.axisgrid")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Pair plots
sns.pairplot(iris_df, hue='species', height=2.5)
plt.suptitle('Pairwise Relationships between Features', y=1.02)
plt.show()


# In[17]:


# Box plots
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal_length', data=iris_df)
plt.title('Sepal Length Distribution by Species')
plt.show()


# Pair plots and box plots provide insights into the relationships and distributions of features across different species.

# In[18]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='sepal_width', data=iris_df)
plt.title('Sepal Width Distribution by Species')
plt.show()


# In[19]:


# Heatmap to visualize correlation between features (excluding 'species' column)
plt.figure(figsize=(8, 6))
sns.heatmap(iris_df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# The correlation heatmap shows strong positive correlations between petal length and petal width, indicating that they tend to increase together.

# Overall, this analysis provides a comprehensive understanding of the Iris dataset and demonstrates the effectiveness of the Random Forest classifier for species classification based on flower measurements.
