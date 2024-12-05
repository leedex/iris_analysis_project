# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load Dataset
try:
    # Load the Iris dataset
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    # Display the first few rows
    print("First few rows of the dataset:")
    print(data.head())
except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Check data structure and missing values
print("\nDataset Info:")
print(data.info())
print("\nMissing Values Check:")
print(data.isnull().sum())

# Task 1: Clean the Dataset
# No missing values in Iris dataset, but if there were:
# data = data.fillna(data.mean())

# Task 2: Basic Data Analysis
# Basic statistics
print("\nBasic Statistics:")
print(data.describe())

# Grouping: Mean of numerical columns by species
grouped_means = data.groupby('species').mean()
print("\nGrouped Means by Species:")
print(grouped_means)

# Task 3: Data Visualization
# Set a seaborn style
sns.set(style="whitegrid")

# 1. Line Chart (mock trend: using cumulative sum as there's no time column in Iris)
data['sepal length (cm) cumulative'] = data['sepal length (cm)'].cumsum()
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['sepal length (cm) cumulative'], label='Cumulative Sepal Length')
plt.title('Cumulative Sepal Length Trend')
plt.xlabel('Index')
plt.ylabel('Cumulative Sepal Length (cm)')
plt.legend()
plt.show()

# 2. Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='sepal length (cm)', data=data, estimator=np.mean, ci=None)
plt.title('Average Sepal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length (cm)')
plt.show()

# 3. Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data['sepal width (cm)'], bins=20, kde=True, color='blue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', style='species', data=data)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Insights
print("\nInsights:")
print("- Setosa species tends to have shorter petals compared to others.")
print("- Virginica species generally has the largest petal and sepal lengths.")
print("- The distribution of sepal width appears roughly normal.")
print("- Cumulative trends can provide a mock visualization for analysis over indices.")
