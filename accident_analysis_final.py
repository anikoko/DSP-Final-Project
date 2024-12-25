import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway

# Load the data
datafile = "Final_test/Accident_Information.csv"
df = pd.read_csv(datafile, low_memory=False)

df = df.head(1000000)

# Task 5: Remove rows where road class is unclassified
df = df[df['1st_Road_Class'] != 'Unclassified']
df = df[df['2nd_Road_Class'] != 'Unclassified']

# print(df.head())

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
df['InScotland'] = df['InScotland'].map({'Yes': 1, 'No': 0})
#
# Task 4: Display the first few rows and summarize the data
print('HEAD')
print(df.head())
print('\nINFO')
print(df.info())
print("\nSUMMARY STATISTICS")
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
# print(df)

print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(list(df.columns))

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
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)  # Add dummies without removing the original column
    else:  # Use label encoding for high cardinality
        df[f"{col}_encoded"] = df[col].astype('category').cat.codes  # Create a new column for encoded values
# print(df)
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(list(df.columns))

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
df['Time_of_Day'] = df['Time_encoded'].apply(get_time_of_day)

# print(df)
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

print(list(df.columns))

# Task 10: Calculate accident density by local authority
accident_density = df['Local_Authority_(District)'].value_counts().reset_index()
accident_density.columns = ['Local_Authority_(District)', 'Accident_Count']

# Output accident density
print('Accident Density')
print(accident_density)


# Task 2.1: Calculate summary statistics for numerical and categorical variables
numerical_summary = df.describe().T
categorical_summary = df.describe(include=['object', 'category']).T

print("Numerical Summary:\n", numerical_summary)
print("Categorical Summary:\n", categorical_summary)
print('HEAD\n')
print(df.head())
print('\nCOLUMNS')
print(list(df.columns))

# Task 2.2: Identify trends in accident severity, time, weather, and road conditions
severity_trend = df.groupby('Accident_Severity')['Accident_Index'].count()
time_trend = df.groupby(df['Time_of_Day'])['Accident_Index'].count()
weather_trend = df.groupby('Weather_Conditions')['Accident_Index'].count()
road_condition_trend = df.groupby('Road_Surface_Conditions')['Accident_Index'].count()

print("Severity Trend:\n", severity_trend)
print("Time Trend:\n", time_trend)
print("Weather Trend:\n", weather_trend)
print("Road Condition Trend:\n", road_condition_trend)

# Task 2.3: Histogram of Speed_limit to understand its distribution
plt.figure(figsize=(8, 6))
df['Speed_limit'].hist(bins=20, edgecolor='black')
plt.title("Distribution of Speed Limit")
plt.xlabel("Speed Limit")
plt.ylabel("Frequency")
plt.show()

# Task 2.4: Bar plot of Day_of_Week vs. number of accidents
plt.figure(figsize=(8, 6))
day_of_week_accidents = df['Day_of_Week'].value_counts()
day_of_week_accidents.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Number of Accidents by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.show()

# Task 2.5: Scatter plot showing Latitude and Longitude for accident locations
plt.figure(figsize=(10, 8))
plt.scatter(df['Longitude'], df['Latitude'], alpha=0.5, c='red', s=1)
plt.title("Accident Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Task 2.5 on a map
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Create a map using Cartopy
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})

# Set the extent of the map (adjust based on your data)
ax.set_extent([-10, 2, 49, 61])  # [lon_min, lon_max, lat_min, lat_max]

# Add map features (coastlines, borders, etc.)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.LAKES, edgecolor='black')
ax.add_feature(cfeature.RIVERS)

# Plot longitude and latitude from the DataFrame
ax.scatter(df['Longitude'], df['Latitude'], color='red', s=0.1, label='Locations', transform=ccrs.PlateCarree())

# Add a title and legend
ax.set_title('Locations on Map', fontsize=16)
ax.legend()

# Show the plot
plt.show()

# Task 2.6: Heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
# Select only numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns
correlation_matrix = df[numerical_columns].corr()

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Task 2.7: Pie chart showing the proportion of accidents by Urban_or_Rural_Area
urban_rural_counts = df['Urban_or_Rural_Area'].value_counts()
plt.figure(figsize=(8, 6))
urban_rural_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
plt.title("Accidents by Urban or Rural Area")
plt.ylabel("")  # Hide the y-axis label
plt.show()

# Task 2.8: Perform t-tests or ANOVA for numerical variables grouped by Accident_Severity
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
anova_results = {}
ttest_results = {}

# Perform ANOVA for variables with more than two groups
for col in numerical_columns:
    groups = [group[col].dropna() for _, group in df.groupby('Accident_Severity')]
    if len(groups) > 2:  # Use ANOVA for more than 2 groups
        anova_results[col] = f_oneway(*groups)
    elif len(groups) == 2:  # Use t-test for exactly 2 groups
        ttest_results[col] = ttest_ind(groups[0], groups[1], equal_var=False)

# Output results
print("ANOVA Results (More than 2 groups):\n")
for col, result in anova_results.items():
    print(f"{col}: F-statistic = {result.statistic:.2f}, p-value = {result.pvalue:.4f}")

print("\nT-Test Results (Exactly 2 groups):\n")
for col, result in ttest_results.items():
    print(f"{col}: t-statistic = {result.statistic:.2f}, p-value = {result.pvalue:.4f}")


# Decision Trees
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import graphviz
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month
df['Day'] = pd.to_datetime(df['Date']).dt.day
df.drop(columns=['Date'], inplace=True)
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
df.drop(columns=['Time'], inplace=True)

# Target and feature selection
target = "Accident_Severity"
features = [
    '1st_Road_Class', '2nd_Road_Class', 'Carriageway_Hazards', 'Day_of_Week',
    'Junction_Control', 'Light_Conditions', 'Road_Surface_Conditions',
    'Road_Type', 'Speed_limit', 'Weather_Conditions'
]

y = df[target]

# Encode categorical variables if necessary (simplified encoding example)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")

# Handle high-cardinality columns
high_card_cols = [col for col in categorical_cols if df[col].nunique() > 100]
low_card_cols = [col for col in categorical_cols if col not in high_card_cols]

# Encode high-cardinality columns with frequency encoding
for col in high_card_cols:
    freq_encoding = df[col].value_counts().to_dict()
    df[col] = df[col].map(freq_encoding)

# One-hot encode low-cardinality columns
df = pd.get_dummies(df, columns=low_card_cols, drop_first=True)

# Split dataset
X = df.drop(columns=['Accident_Severity_Serious', 'Accident_Severity_Slight'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Visualization of the Decision Tree
def visualize_tree(model, feature_names):
    dot_data = export_graphviz(
        model, out_file=None, feature_names=feature_names, class_names=model.classes_.astype(str),
        filled=True, rounded=True, special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")  # Saves as decision_tree.pdf
    return graph

# Generate the interactive decision tree visualization
visualize_tree(clf, X.columns).view()

# Feature Importance Analysis
importance = clf.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Decision Tree")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Feature selection based on threshold
threshold = 0.1
selected_features = feature_importance_df[feature_importance_df["Importance"] > threshold]["Feature"].tolist()
selected_indices = [X.columns.get_loc(feature) for feature in selected_features]

# Train new model with selected features
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]
clf_selected = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_selected.fit(X_train_selected, y_train)

# Evaluate models
print("Original Model Performance:\n", classification_report(y_test, clf.predict(X_test)))
print("Reduced Model Performance:\n", classification_report(y_test, clf_selected.predict(X_test_selected)))
