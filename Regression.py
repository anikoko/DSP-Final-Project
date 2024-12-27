from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, ConfusionMatrixDisplay

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 



class Regression:
    # Linear Regression
    @staticmethod
    def linearRegression(df, features, target):
        X = df[features]
        y = df[target]

        # Split data into training (80%) and test (20%) sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2_mean, mae_mean = Regression.evaluate_model(model, X, y)
        results = {'R-squared': r2_mean, 'Mean Absolute Error': -mae_mean}

        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}") 

        print('LinearRegression results')
        print(results)

        Regression.plot_pred_VS_actual(y_test, y_pred, target)

    @staticmethod
    def evaluate_model(model, X, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        # R-squared
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        # MAE
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        mae_scores = cross_val_score(model, X, y, cv=kf, scoring=mae_scorer)
        
        return np.mean(r2_scores), np.mean(mae_scores)
    
    @staticmethod
    def plot_pred_VS_actual(y_test, y_pred, target):
        # Scatter plot for Actual vs Predicted
        plt.figure(figsize=(12, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, color='b')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--', label='Perfect Prediction')

        plt.title('Actual vs Predicted Values')
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def randomForestRegressor(df, features, target):
        X = df[features]
        y = df[target]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42) 

        # Fit the model to the training data
        rf_model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = rf_model.predict(X_test)

        # Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}") 

        r2_mean, mae_mean = Regression.evaluate_model(rf_model, X, y)
        results = {'R-squared': r2_mean, 'Mean Absolute Error': -mae_mean}

        print('RandomForestRegression')
        print(results)

        # Feature importance
        feature_importance = pd.DataFrame({'Feature': X.columns, 
                                        'Importance': rf_model.feature_importances_})
        feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
        print("\nFeature Importance:")
        print(feature_importance)



    # target = 'InScotland'
    @staticmethod
    def logisticRegression(df, target):
        # Logistic Regression

        # Get features and target
        X = df.drop(target, axis=1)
        y = df[target]

        y = y.fillna(0) 

        
        X, y_train, y_test, X_train_scaled, X_test_scaled, model = Regression.create_model(df, X, y)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        print("Model Performance:")
        print(classification_report(y_test, y_pred))

        # Feature importance
        Regression.feature_importance_logistic(df, X, model)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # visualization
        Regression.visualize_logistic(df, y_test, X_test_scaled, model, y_pred)

    @staticmethod
    def visualize_logistic(df, y_test, X_test_scaled, model, y_pred):
        # Calculate and visualize confusion matrix.
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()

        # Compute precision, recall, and F1 score.
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

        # Plot ROC curve and calculate AUC score
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        print(f'AUC Score: {roc_auc:.4f}')

    @staticmethod
    def create_model(df, X, y):
        # Remove columns with NaN values
        cols_with_nan = X.columns[X.isnull().any()].tolist()
        X = X.drop(columns=cols_with_nan)

        # Sample if needed
        sample_size = 100000
        if len(X) > sample_size:
            np.random.seed(42)
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X.iloc[indices]
            y = y.iloc[indices]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1, random_state=42)
        return X,y_train,y_test,X_train_scaled,X_test_scaled,model

    @staticmethod
    def feature_importance_logistic(df, X, model):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': abs(model.coef_[0])
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))