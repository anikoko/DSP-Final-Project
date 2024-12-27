# DSP-Final-Project
The Final Project to analyse Britain's accident reports.

two main files: one is a simple python file, the other is a jupiter notebook for easy demo purposes


Notes:
1. the data was too large to be uploaded, pleas,see the teams task to get the file
2. accident_analysis_final.py and pandas_tasks.py - are initial drafts and can be ignored.

Contents and tasks:
preprocessing.py
1.	Pandas
1)	Make sure all string-numbers are transformed into numerical columns 
2)	Transform date to pd.date and Time to the time value.
3)	Transform InScotland to Boolean values
4)	Display the first few rows and summarize the data using .info() and .describe().
5)	Remove row if road-class is unclassified.
6)	For numerical variables: Impute with mean/median or flag them as missing.
7)	Detect outliers with statistical technique (IQR) from numerical columns.
8)	Convert categorical variables (e.g., Accident_Severity, Day_of_Week, etc.) into numerical format using one-hot encoding or label encoding.
9)	Create derived features, e.g., Time_of_Day (Morning, Afternoon, Evening, Night) from the Time column.
10)	Calculate accident density by local authority using the Local_Authority_(District) column.
11)	Remove NaN values

2.	More Pandas
1)	Calculate summary statistics for all numerical and categorical variables.
2)	Identify trends in accident severity, time, weather, and road conditions. 
3)	Histogram of Speed_limit to understand its distribution.
4)	Bar plot of Day_of_Week vs. number of accidents.
5)	Scatter plot showing Latitude and Longitude for accident locations.
6)	Heatmap of the correlation matrix to identify relationships between numerical features.
7)	Pie chart showing the proportion of accidents by Urban_or_Rural_Area.
8)	Perform t-tests or ANOVA for numerical variables grouped by Accident_Severity


3. Machine Learning

Decision_tress.py
1.	Objective: Create an interactive visualization of a decision tree.
 Instructions: Build a decision tree classifier on a simple dataset.
 Use graphviz or matplotlib to visualize the tree structure. 
Add color coding for different decision paths. 
Implement node information display on hover.
 Expected Outcome: Interactive tree visualization with clear decision paths and node details. 
2.	 Feature Importance Analysis 
Objective: Analyze and visualize feature importance in decision trees.
 Instructions: Calculate feature importance scores using the trained model. 
Create a bar plot of feature importance rankings.
Implement feature selection based on importance thresholds. 
Compare model performance before and after feature selection. 
Expected Outcome: Ranked feature importance with performance impact analysis. 
Target variable: accident_severity


Regressio.py
Logistic Regression
1.  
Basic Data Preparation and Model Training Objective: Set up a basic logistic regression pipeline with proper data preprocessing. Instructions: 
Load the generated binary classification dataset and display first 5 rows. 
Split data into training (80%) and testing (20%) sets. 
Apply StandardScaler for feature scaling. 
Train a basic logistic regression model and calculate accuracy score.
 Expected Outcome: Working logistic regression pipeline with baseline accuracy metrics. 
2. 
 Evaluation Metrics Implementation Objective: Implement and analyze multiple evaluation metrics for model performance. Instructions: 
Using the model from Task 1, generate predictions on test data. 
Calculate and visualize confusion matrix. 
Compute precision, recall, and F1 score.
 Plot ROC curve and calculate AUC score. 
Expected Outcome: Comprehensive model evaluation with visualizations and metric analysis.
Target variable: InScotland 

Linear Regression
Split data into training (80%) and test (20%) sets.
Calculate and evaluate R-squared and Mean Absolute Error.
1. Predicting Accident Severity (Target: Accident_Severity)
The accident severity may depend on multiple factors like speed limit, weather conditions
2. RandomForestRegresso
to do non-linear regression as the relationship may not be linear

clustering.py
K-means clustering
Use longitude and Latitude to create accident segments with K-Means clustering.
Experiment with values of k from 2 to 5 and use the elbow method to determine the optimal number of clusters.
Plot the clusters.


