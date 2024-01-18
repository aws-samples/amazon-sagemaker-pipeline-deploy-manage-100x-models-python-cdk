import tarfile
import logging
import os
import numpy as np
import pandas as pd

from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from helper import *


if __name__ == "__main__":
    print("=============================")
    print("Starting model evaluation job")
    print("=============================")
    
    xgb_model = get_xgboost_model()
    print("Loading test input data")

    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    
    X_test = xgb.DMatrix(df.values)
    
    print('Performing predictions against test data.')
    predictions = xgb_model.predict(X_test)
    # predicted_probs = xgb_model.predict_proba(X_test)
        
    metrics_table = get_metrics_table(predictions, y_test, n_bins=1) ### As data is very small for demo purpose so kept n_bin=1. But can be increased to higher number.

    output_path = f"/opt/ml/processing/post_training_analysis"
    metrics_table.to_csv(f"{output_path}/post_training_analysis.csv", index=False)

    # Feature importance
    output_path_feature_importance = "/opt/ml/processing/feature_importance"

    features_imp_dict = xgb_model.get_score(importance_type='weight')
    features_imp_df = pd.DataFrame(list(features_imp_dict.items()), columns=['Feature', 'Value'])
    features_imp_df.to_csv(
        f"{output_path_feature_importance}/feature_importance.csv", index=False
    )

    print("=============================")
    print("Model evaluation job completed")
    print("=============================")
