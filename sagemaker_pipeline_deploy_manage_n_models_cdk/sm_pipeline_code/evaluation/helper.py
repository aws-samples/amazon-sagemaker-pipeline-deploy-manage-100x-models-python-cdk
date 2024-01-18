import json
import logging
import os
import tarfile
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, roc_auc_score)


def is_safe_member(member):
    # Implement your validation logic here
    if member.name=="xgboost-model":
        return True
    
    
def get_xgboost_model():
    model_dir = '/opt/ml/processing/model'

    model_tar_gz_path = os.path.join(model_dir, 'model.tar.gz')
    with tarfile.open(model_tar_gz_path) as tar:
        for member in tar.getmembers():
            if is_safe_member(member):
                tar.extract(member, path=model_dir)

    print("load the model")
    
    xgb_model = xgb.Booster()
    model_file_path = os.path.join(model_dir, "xgboost-model")
    xgb_model.load_model(model_file_path)
    return xgb_model
