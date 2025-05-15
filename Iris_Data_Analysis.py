
# Iris Data Analysis and Visualization Assignment

# Task 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the dataset using sklearn
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully.\n")
except Exception as e:
    print("Error loading dataset:", e)

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Cleaning (if necessary)
df = df.dropna()  # Drop rows with missing values (none in Iris)

# Task 2: Basic Data Analysis

# Describe the dataset
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute mean
grouped = df.groupby("species").mean()
print("\nAverage values by species:")
print(grouped)

# Observations
print("\nObservations:")
print("1. Iris-virginica has the highest average petal length and width.")
print("2. Iris-setosa has shorter petals and sepals compared to others.")
print("3. There's a clear distinction between species based on petal dimensions.")

# Task 3: Data Visualization

# 1. Line Chart: Mean petal length over species (not a real time-series but for demo)
plt.figure(figsize=(6, 4))
grouped["petal length (cm)"].plot(kind="line", marker="o")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Bar Chart: Sepal Width per Species
plt.figure(figsize=(6, 4))
grouped["sepal width (cm)"].plot(kind="bar", color="skyblue")
plt.title("Average Sepal Width per Species")
plt.xlabel("Species")
plt.ylabel("Sepal Width (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram: Distribution of Petal Length
plt.figure(figsize=(6, 4))
plt.hist(df["petal length (cm)"], bins=20, color="orange", edgecolor="black")
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter Plot: Sepal Length vs. Petal Length
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species")
plt.title("Sepal Length vs. Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()
