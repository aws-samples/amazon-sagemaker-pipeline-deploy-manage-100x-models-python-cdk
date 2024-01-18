import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, roc_auc_score)
from helper import *

if __name__ == "__main__":
    print('Starting evaluation.')
    
    xgb_model = get_xgboost_model()
    print("Loading test input data")

    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    
    X_test = xgb.DMatrix(df.values)
    
    print('Performing predictions against test data.')
    predictions = xgb_model.predict(X_test)

    print("Creating classification evaluation report")
    acc = accuracy_score(y_test, predictions.round())
    auc = roc_auc_score(y_test, predictions.round())
    std = np.std(y_test - predictions)
    
    # The metrics reported can change based on the model used, 
    # but it must be a specific name per 
    # (https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {
                "value": acc,
                "standard_deviation": std,
            },
            "auc": {"value": auc, "standard_deviation": std},
        },
    }
  
    print("Classification report:\n{}".format(report_dict))
    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
