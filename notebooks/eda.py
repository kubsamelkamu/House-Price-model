import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data
data = pd.read_csv('./data/processed_train.csv')

# Display the first few rows of the dataset
print(data.head())

# Generate descriptive statistics
print(data.describe())

# Visualize the distribution of the target variable (SalePrice)
plt.figure(figsize=(10, 6))
sns.histplot(data['SalePrice'], kde=True, bins=30)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

# Visualize the distribution of other numeric features
numeric_features = data.select_dtypes(include=['number']).columns
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Scatter plot of SalePrice vs. numeric features
for feature in numeric_features:
    if feature != 'SalePrice':
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data[feature], y=data['SalePrice'])
        plt.title(f'SalePrice vs. {feature}')
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()

# Pair plot of top features
top_features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
sns.pairplot(data[top_features])
plt.show()

# Correlation matrix
plt.figure(figsize=(15, 10))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()