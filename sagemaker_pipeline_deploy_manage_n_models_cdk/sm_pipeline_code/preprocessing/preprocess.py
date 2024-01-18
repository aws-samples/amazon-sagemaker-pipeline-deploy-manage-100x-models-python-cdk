import pandas as pd
import numpy as np
import argparse
import os
import os
import glob

def read_parameters():
    """
    Read job parameters
    Returns:
        (Namespace): read parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dim', type=str, default='CA')
    params, _ = parser.parse_known_args()
    return params

# Helper method
def process(churn):
    # Add two new indicators
    churn = churn.drop("Phone", axis=1)
    churn["Area Code"] = churn["Area Code"].astype(object)

    churn = churn.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

    model_data = pd.get_dummies(churn)
    model_data = pd.concat(
        [model_data["Churn?_True."], model_data.drop(["Churn?_False.", "Churn?_True."], axis=1)], axis=1
    )
    model_data = model_data.astype(float)
    model_data.iloc[:, 0]= model_data.iloc[:, 0].astype(int)  ## if you don't convert the lables to integer then Model quality job will consider 0, 1, 1.0. 0.0 as different labels.
    return model_data

print(f"===========================================================")
print(f"Starting pre-processing")
print(f"Reading parameters")

# reading job parameters
args = read_parameters()
print(f"Parameters read: {args}")
state_dim =args.state_dim
print(f"dummy dimesnion read is :{state_dim}")
print("******************************")

# set input and output paths
input_data_path = "/opt/ml/processing/input"
csv_files = glob.glob(os.path.join(input_data_path, '*.csv'))
if len(csv_files) == 0:
    print("No CSV files found in the directory.")
elif len(csv_files) == 1:
    # Read the first (and only) CSV file found
    input_data_path = csv_files[0]
else:
    print("Custom Error: Multiple CSV files found in the directory. Specify which one to read.")
    
train_data_path = "/opt/ml/processing/training"
val_data_path = "/opt/ml/processing/validation"
test_data_path = "/opt/ml/processing/test"

try:
    os.makedirs(train_data_path)
    os.makedirs(val_data_path)
    os.makedirs(test_data_path)
except:
    print("Directories were not created")


df = pd.read_csv(input_data_path, engine='python')
df = df[df["Dimension"] == state_dim]
df = process(df)
    
df_train, df_validation, df_test = np.split(df.sample(frac=1, random_state=1729),
                                            [int(0.7 * len(df)), int(0.9 * len(df))],
                                        )

print("df_train****")
print(df_train.info())
print("-------------------------------------")
print("df_validation****")
print(df_validation.info())
print("-------------------------------------")
print("df_test****")
print(df_test.info())
print("-------------------------------------")


print("Saving csv data")
df_train.to_csv(train_data_path+'/train.csv', index=False, header=False)
df_test.to_csv(val_data_path+'/validation.csv',  index=False, header=False)
df_test.to_csv(test_data_path+'/test.csv',  index=False, header=False)

print(f"Ending pre-processing")
print(f"===========================================================")