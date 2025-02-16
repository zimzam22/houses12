import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

house = pd.read_csv("D:\\Assignment\\CPSC_4800\\houses.csv")

# Get a concise summary of the dataframe
summary = house.describe()

# Get information about the dataframe (structure, data types, non-null counts)
structure = house.info()

# Display the summary and structure
print("Summary of the dataset:")
print(summary)
print("\nStructure of the dataset:")
print(structure)

house['LotFrontage'] = house['LotFrontage'].fillna(house['LotFrontage'].mean())

# Number of columns to check at a time
chunk_size = 20

# Calculate the total number of columns
num_columns = house.shape[1]

# Loop through the columns in chunks of 20
for start in range(0, num_columns, chunk_size):
    # Select the next chunk of 20 columns
    chunk = house.iloc[:, start:start + chunk_size]
    
    # Check for missing values in this chunk
    missing_values_chunk = chunk.isnull().sum()
    
    # Print the missing values for the current chunk
    print(f"Missing values for columns {start + 1} to {min(start + chunk_size, num_columns)}:")
    print(missing_values_chunk)
    print("\n")
	
	
# List of columns for which you want to impute the mode
columns_to_impute = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
                     'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 
                     'GarageCond', 'Fence', 'MiscFeature', 'Alley','PoolQC']

# Loop through each column and fill missing values with the mode
for column in columns_to_impute:
    mode_value = house[column].mode()[0]  # Find the mode for the column
    house[column] = house[column].fillna(mode_value)  # Impute missing values with the mode (no inplace=True)
	
# Number of columns to check at a time
chunk_size = 20

# Calculate the total number of columns
num_columns = house.shape[1]

# Loop through the columns in chunks of 20
for start in range(0, num_columns, chunk_size):
    # Select the next chunk of 20 columns
    chunk = house.iloc[:, start:start + chunk_size]
    
    # Check for missing values in this chunk
    missing_values_chunk = chunk.isnull().sum()
    
    # Print the missing values for the current chunk
    print(f"Missing values for columns {start + 1} to {min(start + chunk_size, num_columns)}:")
    print(missing_values_chunk)
    print("\n")
	
# List of columns to convert to categorical
categorical_columns = [
    'MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'MoSold', 'YrSold','HalfBath','GarageCars'
]

# Convert each column to categorical
for column in categorical_columns:
    house[column] = house[column].astype('category')
	
summary = house.describe()

print("Summary Statistics for Numerical Columns:")
print(summary)

# List of columns to check for outliers
columns_to_check = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
    'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice'
]

# Function to calculate outliers based on IQR
def find_outliers(df, columns):
    outlier_counts = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identifying outliers and counting them
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = outliers.shape[0]
    
    return outlier_counts

# Call the function
outlier_counts = find_outliers(house, columns_to_check)

# Display the outlier counts for each column
for col, count in outlier_counts.items():
    print(f"Outliers for column '{col}': {count}")
	
# Plot box plots for the specified columns
plt.figure(figsize=(20, 12))

# Set a seaborn style for better aesthetics
sns.set(style="whitegrid")

# Create a boxplot for each column
for i, col in enumerate(columns_to_check, 1):
    plt.subplot(5, 4, i)  # 5 rows and 4 columns of subplots
    sns.boxplot(x=house[col])
    plt.title(f'Box plot for {col}')
    plt.tight_layout()

# Show all the box plots
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Get categorical columns to plot (excluding specified columns)
columns_to_check = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
    'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice'
]

# Get categorical columns to plot (excluding specified columns)
other_columns = [col for col in house.columns if col not in columns_to_check and house[col].dtype == 'object']

# If you have too many categorical columns, limit the number
other_columns = other_columns[:10]  # Limit to first 10 categorical columns, you can change this

# Set up the grid size to match the number of remaining columns
num_plots = len(other_columns)
cols = 4  # Number of columns for the subplot grid
rows = (num_plots // cols) + (num_plots % cols > 0)  # Dynamic number of rows to fit the plots

# Plot bar graphs for the other columns
plt.figure(figsize=(20, 5 * rows))  # Adjust figure height to fit all plots

# Set a seaborn style for better aesthetics
sns.set(style="whitegrid")

# Create a bar plot for each of the remaining columns
for i, col in enumerate(other_columns, 1):
    plt.subplot(rows, cols, i)  # Dynamically determine row and column position
    sns.countplot(x=house[col])
    plt.title(f'Bar plot for {col}')
    
    # If it's a column with many categories like "neighborhood", rotate the labels
    if house[col].dtype == 'object' and len(house[col].unique()) > 20:  # You can adjust the threshold
        plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility

    plt.tight_layout()

# Show all the bar plots
plt.show()


cols = ['SalePrice', 'LotArea', 'BldgType', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
        'BsmtFullBath', 'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'GarageCars']

# Select only the columns needed from the dataset
df = house[cols]

# 1. Correlation Matrix for numerical variables (BldgType is categorical, so we exclude it)
numerical_cols = ['SalePrice', 'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
                  'BsmtFullBath', 'BsmtHalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'GarageCars']

# Compute the correlation matrix
corr_matrix = df[numerical_cols].corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# 2. Scatter plots for numerical variables
# Plot scatter plots for each pair of numerical variables
sns.pairplot(df[numerical_cols])
plt.suptitle("Scatter Plots of Numerical Variables", y=1.02)
plt.show()