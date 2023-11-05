import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

# Load the filtered_data
macro_data = pd.read_csv('/Users/ced/Documents/PolyHacks/Data/Macro_datas.csv', parse_dates=['Date'])
industry_data = pd.read_csv('/Users/ced/Documents/PolyHacks/Data/benchmark/12_Industry_Portfolios.CSV', parse_dates=['Date'])

# Merge the datasets on the date field
merged_data = pd.merge_asof(industry_data.sort_values('Date'), 
                            macro_data.sort_values('Date'), 
                            on='Date', 
                            direction='backward')

# Drop any rows with missing values after the merge
merged_data.dropna(inplace=True)

# Define the feature columns (assuming the first column after 'Date' in macro_data is a feature)
feature_columns = macro_data.columns.difference(['Date']).tolist()

# Define performance classification based on median daily_returns
def classify_performance(return_value, median_value):
    return 'High Performance' if return_value > median_value else 'Low Performance'

# Function to perform classification and evaluation for a given industry
def classify_and_evaluate(industry_name, train_data, test_data, features):
    # Calculate the median value from the training filtered_data
    median_value = train_data[industry_name].median()
    
    # Classify performance for both training and testing sets
    y_train = train_data[industry_name].apply(classify_performance, args=(median_value,))
    y_test = test_data[industry_name].apply(classify_performance, args=(median_value,))
    
    # Encode the classified performance
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Train the Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    random_search_rf = RandomizedSearchCV(
        estimator=rf_classifier,
        param_distributions=param_grid_rf,
        n_iter=15,
        cv=5,
        scoring='f1_macro',
        random_state=42,
        n_jobs=-8
    )
    random_search_rf.fit(train_data[features], y_train_encoded)
    best_classifier = random_search_rf.best_estimator_
    
    # Evaluate the classifier on the test set
    y_pred_test = best_classifier.predict(test_data[features])
    f1 = f1_score(y_test_encoded, y_pred_test, average='macro')
    report = classification_report(y_test_encoded, y_pred_test)
    
    return f1, report

# Split the filtered_data based on the date
train_data = merged_data[merged_data['Date'].dt.year.between(1995, 1999)]
test_data = merged_data[merged_data['Date'].dt.year.between(2000, 2015)]

# Apply the classification and evaluation to each industry
results = {}
industry_columns = industry_data.columns.difference(['Date']).tolist()
for industry in industry_columns:
    f1, report = classify_and_evaluate(industry, train_data, test_data, feature_columns)
    results[industry] = {'F1 Score': f1, 'Classification Report': report}

# Output the results
for industry, result in results.items():
    print(f'Industry: {industry}')
    print(f'F1 Score: {result["F1 Score"]}')
    print(result["Classification Report"])
    print('-----------------------------------------------------')
