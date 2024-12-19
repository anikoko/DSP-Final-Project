import pandas as pd
import numpy as np

# Load the data
datafile = "Accident_Information.csv"
df = pd.read_csv(datafile, low_memory=False)


# Task 5: Remove rows where road class is unclassified
df = df[df['1st_Road_Class'] != 'Unclassified']
df = df[df['2nd_Road_Class'] != 'Unclassified']

print(df.head())

# Task 1: Transform string-numbers into numerical columns
columns_to_convert = [
    "Accident_Index",
    "1st_Road_Number",
    "2nd_Road_Number",
    "Did_Police_Officer_Attend_Scene_of_Accident",
    "Latitude",
    "Location_Easting_OSGR",
    "Location_Northing_OSGR",
    "Longitude",
    "Number_of_Casualties",
    "Number_of_Vehicles",
    "Pedestrian_Crossing-Human_Control",
    "Pedestrian_Crossing-Physical_Facilities",
    "Speed_limit",
    "Year",
]
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Task 2: Transform 'Date' to datetime and 'Time' to time value
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.time

# Task 3: Transform 'InScotland' to Boolean values
df['InScotland'] = df['InScotland'].map({'Yes': True, 'No': False})
#
# Task 4: Display the first few rows and summarize the data
print(df.head())
print(df.info())
print(df.describe(include='all'))

# Task 6: Impute missing values in numerical variables
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    if df[col].isnull().sum() > 0:
        mean_or_median = 'mean' if df[col].skew() < 1 else 'median'
        if mean_or_median == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].median())

# Task 7: Detect outliers using IQR
outliers = {}
for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index

# Task 8: Convert categorical variables to numerical format
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if df[col].nunique() <= 10:  # Use one-hot encoding for low cardinality
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    else:  # Use label encoding for high cardinality
        df[col] = df[col].astype('category').cat.codes

# Task 9: Create derived feature 'Time_of_Day'
def get_time_of_day(time):
    # Convert military time integer to hour
    hour = int(str(time).zfill(4)[:2])  # Extract the first two digits (hours)

    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'


# Apply the function
df['Time_of_Day'] = df['Time'].apply(get_time_of_day)

print(df)
print(list(df.columns))

# Task 10: Calculate accident density by local authority
accident_density = df['Local_Authority_(District)'].value_counts().reset_index()
accident_density.columns = ['Local_Authority_(District)', 'Accident_Count']

# Output accident density
print(accident_density)
