import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["figure.figsize"] = (20, 10)

# Load dataset
df1 = pd.read_csv("bengaluru_house_prices.csv")

print("First 5 rows:")
print(df1.head())

print("\nDataset Shape:")
print(df1.shape)

# Drop unnecessary columns
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')

print("\nAfter Dropping Columns:")
print(df2.head())

# Remove null values
df3 = df2.dropna()

print("\nNull Values Removed:")
print(df3.isnull().sum())

# Add BHK column
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

print("\nBHK Added:")
print(df3.head())

# Function to check float values
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# Show invalid sqft values
print("\nInvalid total_sqft values:")
print(df3[~df3['total_sqft'].apply(is_float)].head(10))

# Convert sqft ranges to average
def convert_sqft_to_num(x):
    tokens = x.split('-')

    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2

    try:
        return float(x)

    except:
        return None

# Apply conversion
df4 = df3.copy()

df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)

df4 = df4[df4.total_sqft.notnull()]

print("\nConverted total_sqft:")
print(df4.head())

# Add price per sqft
df5 = df4.copy()

df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']

print("\nPrice Per Sqft Added:")
print(df5.head())

# Clean location column
df5.location = df5.location.apply(lambda x: x.strip())

location_stats = df5['location'].value_counts(ascending=False)

print("\nLocation Statistics:")
print(location_stats.head(20))

# Locations with less than or equal to 10 entries
location_stats_less_than_10 = location_stats[location_stats <= 10]

# Replace rare locations with 'other'
df5.location = df5.location.apply(
    lambda x: 'other' if x in location_stats_less_than_10 else x
)

print("\nUnique Locations After Reduction:")
print(len(df5.location.unique()))

# Remove outliers using business logic
df6 = df5[~(df5.total_sqft / df5.bhk < 300)]

print("\nAfter Removing BHK Outliers:")
print(df6.shape)

# Remove price per sqft outliers
def remove_pps_outliers(df):
    df_out = pd.DataFrame()

    for key, subdf in df.groupby('location'):

        m = np.mean(subdf.price_per_sqft)

        st = np.std(subdf.price_per_sqft)

        reduced_df = subdf[
            (subdf.price_per_sqft > (m - st)) &
            (subdf.price_per_sqft <= (m + st))
        ]

        df_out = pd.concat([df_out, reduced_df], ignore_index=True)

    return df_out

df7 = remove_pps_outliers(df6)

print("\nAfter Removing Price Per Sqft Outliers:")
print(df7.shape)

# Save cleaned dataset
df7.to_csv("cleaned_bangalore_home_prices.csv", index=False)

print("\nCleaned dataset saved successfully!")

print("\nFinal Dataset Preview:")
print(df7.head())

print("\nFinal Dataset Info:")
print(df7.info())

# Example plot
plt.hist(df7.price_per_sqft, bins=50)

plt.xlabel("Price Per Sqft")

plt.ylabel("Count")

plt.title("Distribution of Price Per Sqft")

plt.show()