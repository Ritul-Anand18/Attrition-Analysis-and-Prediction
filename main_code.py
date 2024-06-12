# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:07:15 2024

@author: Ritul Anand
"""

# Importing the basic libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost
from sklearn.naive_bayes import GaussianNB

# Importing the dataset
data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
print("First 5 rows of the dataset:")
print(data.head(5))

# Checking the structure and type of values in the data
print("\nDataset information:")
print(data.info())
print("\nNull values in each column:")
print(data.isnull().sum())

# Dropping additional features for initial visualization
features = data.drop(['Attrition', 'EmployeeNumber', 'MonthlyRate'], axis=1)
output = data['Attrition']

# Visualizing the fraction of attrition in each category
fig, axs = plt.subplots(nrows=len(features.columns), ncols=1, figsize=(8, 6 * len(features.columns)))

for i, feature in enumerate(features.columns):
    ax = axs[i]
    fractions = data.groupby(feature)['Attrition'].value_counts(normalize=True).unstack().fillna(0)['Yes']
    fractions.plot(kind='bar', ax=ax)
    ax.set_title(f'Feature: {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Fraction of Attrition (Yes)')

plt.subplots_adjust(hspace=0.5)
plt.show()

# Dropping less useful features
data = data.drop(['StockOptionLevel', 'RelationshipSatisfaction', 'Over18'], axis=1)

# Encoding categorical data using OneHotEncoder
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Attrition')

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_data = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

# Concatenating the encoded features with numerical features
numerical_cols = data.select_dtypes(exclude=['object']).columns.tolist()
data_encoded = pd.concat([data[numerical_cols], encoded_df], axis=1)

# Encoding the target variable
label_encoder = LabelEncoder()
data_encoded['Attrition'] = label_encoder.fit_transform(data['Attrition'])

# Saving the encoded data to a new CSV file
data_encoded.to_csv('encoded_data.csv', index=False)
print("\nEncoded data saved to 'encoded_data.csv'")

# Scaling the features
feature = data_encoded.drop(columns=['Attrition'])
target = data_encoded['Attrition']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature)
scaled_feature_data = pd.DataFrame(scaled_features, columns=feature.columns)
scaled_feature_data = pd.concat([scaled_feature_data, target], axis=1)

# Displaying the correlation matrix with enhanced visualization
plt.figure(figsize=(20, 20))
correlation_matrix = scaled_feature_data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=.5)
plt.title('Correlation Matrix of Features')
plt.show()

# Dropping redundant columns after one-hot encoding
scaled_feature_data = scaled_feature_data.drop(['Gender_Female', 'OverTime_No'], axis=1)

# Splitting the data into training and testing sets
X = scaled_feature_data.drop('Attrition', axis=1)
y = scaled_feature_data['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print(f"\nShape of training set: {X_train.shape}")
print(f"Shape of testing set: {X_test.shape}")

# Logistic Regression
lg = LogisticRegression()
lg.fit(X_train, y_train)
lg_predictions = lg.predict(X_test)

# Accuracy of Logistic Regression
lg_accuracy = accuracy_score(y_test, lg_predictions)
print("\nLogistic Regression")
print(f"Accuracy: {lg_accuracy:.4f}")

# Confusion Matrix for Logistic Regression
confmat = confusion_matrix(y_test, lg_predictions)
print("Confusion Matrix:")
print(confmat)
print("Classification Report:")
print(classification_report(y_test, lg_predictions))

# Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)

# Accuracy of Decision Tree Classifier
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("\nDecision Tree Classifier")
print(f"Accuracy: {dt_accuracy:.4f}")

# Confusion Matrix for Decision Tree Classifier
confmat = confusion_matrix(y_test, dt_predictions)
print("Confusion Matrix:")
print(confmat)
print("Classification Report:")
print(classification_report(y_test, dt_predictions))

# Random Forest Classifier with Grid Search for hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
    'max_depth': [8, 10, None]
}
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("\nRandom Forest Classifier")
print("Best parameters found: ", best_params)

rfc = RandomForestClassifier(random_state=42, **best_params)
rfc.fit(X_train, y_train)
rf_predictions = rfc.predict(X_test)

# Accuracy of Random Forest Classifier
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Accuracy: {rf_accuracy:.4f}")

# Confusion Matrix for Random Forest Classifier
confmat = confusion_matrix(y_test, rf_predictions)
print("Confusion Matrix:")
print(confmat)
print("Classification Report:")
print(classification_report(y_test, rf_predictions))

# Visualizing the confusion matrix
sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest Classifier')
plt.show()

# XGBoost Classifier
params = {
    'learning_rate': [0.05, 0.10, 0.15, 0.20],
    'max_depth': [3, 4, 5, 6, 8, 10, 12],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
}
xgc = xgboost.XGBClassifier()
random_search = RandomizedSearchCV(xgc, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("\nXGBoost Classifier")
print("Best parameters found: ", best_params)

xg = xgboost.XGBClassifier(**best_params)
xg.fit(X_train, y_train)
xg_predictions = xg.predict(X_test)

# Accuracy of XGBoost Classifier
xg_accuracy = accuracy_score(y_test, xg_predictions)
print(f"Accuracy: {xg_accuracy:.4f}")

# Confusion Matrix for XGBoost Classifier
confmat = confusion_matrix(y_test, xg_predictions)
print("Confusion Matrix:")
print(confmat)
print("Classification Report:")
print(classification_report(y_test, xg_predictions))

# Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predictions = nb.predict(X_test)

# Accuracy of Naive Bayes Classifier
nb_accuracy = accuracy_score(y_test, nb_predictions)
print("\nNaive Bayes Classifier")
print(f"Accuracy: {nb_accuracy:.4f}")

# Confusion Matrix for Naive Bayes Classifier
confmat = confusion_matrix(y_test, nb_predictions)
print("Confusion Matrix:")
print(confmat)
print("Classification Report:")
print(classification_report(y_test, nb_predictions))

"""
Conclusion and Further Explorations
Here, the best fitting models were logistic regression and random forest giving accuracy of about 88%. 
We can also look at the performance of each using ROC curves. Further as we tested our model only on test dataset, 
it may be possible that our models were performing poor or good only on these. To test the overall performance we can further divide test set into different sets where one to be used as test and others as training. This will give a better score of our model.
"""
