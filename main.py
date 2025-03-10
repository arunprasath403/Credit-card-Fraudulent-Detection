import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Display dataset information
df.info()
print(df.head())
print(df.isnull().sum())  # Check for missing values
print(df['Class'].value_counts())  # Count fraud vs. non-fraud cases

# Separate legitimate and fraudulent transactions
legit = df[df.Class == 0]
fraud = df[df.Class == 1]

print("Legitimate transactions shape:", legit.shape)
print("Fraudulent transactions shape:", fraud.shape)

# Analyze amount distributions
print("Legitimate transactions amount stats:\n", legit.Amount.describe())
print("Fraudulent transactions amount stats:\n", fraud.Amount.describe())

# Take a random sample of legitimate transactions to balance the dataset
legit_sample = legit.sample(n=492, random_state=2)

# Combine the sampled legitimate transactions with all fraudulent transactions
new_df = pd.concat([legit_sample, fraud], axis=0)

print(new_df['Class'].value_counts())  # Verify class balance
print(new_df.groupby('Class').mean())  # Mean feature values per class

# Split features and target variable
X = new_df.drop(columns='Class', axis=1)
Y = new_df['Class']

# Split into training and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print("Feature matrix shape:", X.shape)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate model accuracy
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data:', test_data_accuracy)

# Save the model (optional)
import joblib
joblib.dump(model, 'credit_fraud_model.pkl')
