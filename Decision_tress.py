import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import graphviz
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

class DesicionTrees():


    @staticmethod
    def decision_trees(df, target, features):
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        df['Month'] = pd.to_datetime(df['Date']).dt.month
        df['Day'] = pd.to_datetime(df['Date']).dt.day
        df.drop(columns=['Date'], inplace=True)
        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
        df.drop(columns=['Time'], inplace=True)

        y = df[target]

        # Encode categorical variables if necessary (simplified encoding example)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        # for col in categorical_cols:
        #     print(f"{col}: {df[col].nunique()} unique values")

        # # Handle high-cardinality columns
        # high_card_cols = [col for col in categorical_cols if df[col].nunique() > 100]
        # low_card_cols = [col for col in categorical_cols if col not in high_card_cols]

        # # Encode high-cardinality columns with frequency encoding
        # for col in high_card_cols:
        #     freq_encoding = df[col].value_counts().to_dict()
        #     df[col] = df[col].map(freq_encoding)


        # One-hot encode low-cardinality columns
        # df = pd.get_dummies(df, columns=low_card_cols, drop_first=True)

        df = df.drop(columns=categorical_cols)
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
        X_test_selected, clf_selected = DesicionTrees.feature_importance_analysis(X, X_train, X_test, y_train, clf)

        # Evaluate models
        print("Original Model Performance:\n", classification_report(y_test, clf.predict(X_test)))
        print("Reduced Model Performance:\n", classification_report(y_test, clf_selected.predict(X_test_selected)))

    @staticmethod
    def feature_importance_analysis(X, X_train, X_test, y_train, clf):
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
        return X_test_selected,clf_selected
    




# Target and feature selection
target = "Accident_Severity"
features = [
    '1st_Road_Class', '2nd_Road_Class', 'Carriageway_Hazards', 'Day_of_Week',
    'Junction_Control', 'Light_Conditions', 'Road_Surface_Conditions',
    'Road_Type', 'Speed_limit', 'Weather_Conditions'
]



