import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from get_data import read_params
import argparse
import joblib
import json



def eval_metrics(actual, pred):
    roc_auc = roc_auc_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    return roc_auc, accuracy


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]


    C = config["estimators"]["LogisticRegression"]["params"]["C"]
    max_iter = config["estimators"]["LogisticRegression"]["params"]["max_iter"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    
    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)


    lr = LogisticRegression(
        C=C, 
        max_iter=max_iter, 
        random_state=random_state)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    
    (roc_auc, accuracy) = eval_metrics(test_y, predicted_qualities)

    print("Logistic Regression model (C=%f, max_iter=%f):" % (C, max_iter))
    print(" roc_auc_score: %s" % roc_auc)
    print("  accuracy_score: %s" % accuracy)

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores={
            "roc_auc_score": roc_auc,
            "accuracy_score": accuracy
        }

        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params={
            "C": C,
            "max_iter": max_iter
        }

        json.dump(params, f, indent=4)


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(lr, model_path)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)