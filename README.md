# Credit Card Fraud Detection

## Overview
This project uses logistic regression to detect fraudulent transactions in a credit card dataset. The dataset is preprocessed, balanced, and used to train a classification model.

## Dataset
The dataset `creditcard.csv` contains transaction details, including various numerical features and a `Class` label (0 = legitimate, 1 = fraud).

You can download the dataset from [Here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Installation

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Run the script to train and test the model:
```sh
main.py
```

## Output
- The model's accuracy on training and test data.
  - **Accuracy on training data:** 0.9517
  - **Accuracy on test data:** 0.9492
- A trained logistic regression model saved as `credit_fraud_model.pkl`.

## Dependencies
- pandas
- numpy
- scikit-learn
- joblib
