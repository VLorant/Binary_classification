import json
import re
from functools import wraps
from io import StringIO
from os import listdir
from typing import Type, Any, Callable

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.base import ClassifierMixin
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, auc, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold


def runtime(func: Callable) -> Callable:
    """
    Time counting wrapper

    Parameters:
        func: function

    Returns:
        function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Runtime of {func.__name__}: {end - start} seconds")
        return result

    return wrapper


@runtime
def search_best_params(u_model: Type[ClassifierMixin], u_params: dict, u_train: str, u_test: list[str], u_features: int,
                       save_file_name: str,
                       raw_output: bool = False, save: bool = False) -> pd.DataFrame:
    """
    To find the best parameters for a classifier

    Parameters:
        u_model: Classifier
        u_params: Dictionary of parameters
        u_train: Path to the training data
        u_test: Path to the test data
        u_features: Number of features
        save_file_name: Path to the output file
        raw_output: Raw output of the model
        save: Save the model

    Returns:
        pd.DataFrame: DataFrame of best parameters
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=20)

    search = GridSearchCV(
        u_model(),
        param_grid=u_params,
        cv=cv,
        n_jobs=-1,
        scoring='f1',
        refit=True
    )

    def read_arff(filename: str) -> pd.DataFrame:
        read_data = ""
        with open(filename, 'r') as f:
            read_data = f.read()
        read_data = read_data[:read_data.index("@inputs")] + read_data[read_data.index("@data"):]
        data = StringIO(read_data)
        data, meta = arff.loadarff(data)
        df = pd.DataFrame(data)
        if 'cluster_idx' in df.columns:
            df = df.drop(['cluster_idx', 'is_noise'], axis=1)
        df['Class'] = df['Class'].astype(int)
        return df

    score_cols: list[str] = ['Accuracy', 'Sensitivity', 'Specificity', 'F1', 'G-mean', 'AUC', 'Balanced Accuracy']
    data = read_arff(u_train)
    X = data.iloc[:, :u_features]
    y = data.iloc[:, u_features]
    search.fit(X, y)
    scores = pd.DataFrame(
        index=[1, 2, 3],
        columns=score_cols,
        dtype=np.float64
    )
    for i in range(1, 4):
        test = read_arff(u_test[i - 1])
        X_test = test.iloc[:, :u_features]
        y_test = test.iloc[:, u_features]
        y_pred = search.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        scores.loc[i] = [
            accuracy_score(y_test, y_pred),
            recall_score(y_test, y_pred, pos_label=1),
            tn / (tn + fp),
            f1_score(y_test, y_pred, pos_label=1),
            np.sqrt(recall_score(y_test, y_pred, pos_label=1) * (tn / (tn + fp))),
            auc(y_test, y_pred),
            balanced_accuracy_score(y_test, y_pred)
        ]
    if not raw_output:
        datas = pd.DataFrame(data=scores.mean(), index=score_cols, columns=['Mean'])
        if save:
            datas.to_json(save_file_name + '.json', orient='split')
        return datas
    if save:
        scores.to_json(save_file_name + '.json', orient='split')
        return scores
    else:
        return scores


@runtime
def automated_file_select_search(train_path: str, test_path: str, u_model: Type[ClassifierMixin], u_params: dict,
                                 save_file_name: str, raw_output: bool = False, save: bool = False):
    """
    Automatically go through all the files and find the best parameters

    Parameters:
        train_path: Path to the training data
        test_path: Path to the test data
        u_model: Classifier
        u_params: Dictionary of parameters
        save_file_name: Path to the output file
        raw_output: Raw output of the model
        save: Save the model

    Returns:
        pd.DataFrame: DataFrame of best parameters
    """

    def sorting_params(f_name: str):
        n = re.search(r'N_(\d+)', f_name)
        d = re.search(r'D_(\d+)', f_name)
        n_min = re.search(r'Nmin_(\d+)', f_name)
        z_min = re.search(r'Zmin_(\d+)', f_name)
        cl = re.search(r'CL_(\d+)', f_name)
        cl2 = re.search(r'_(\d+)-R', f_name)
        return int(n.group(1)), int(d.group(1)), int(n_min.group(1)), int(z_min.group(1)), int(cl.group(1)), int(
            cl2.group(1))

    train_files = listdir(train_path)
    train_files.sort(key=sorting_params)
    train_and_test_dict: dict[str, list[str]] = {}
    for data in train_files:
        train_and_test_dict[train_path + "/" + data] = [
            test_path + "/" + data.replace(data[data.index("Zmin"):data.index(".dat")], "Zmin_0-CL_1_1-R_0_test_1"),
            test_path + "/" + data.replace(data[data.index("Zmin"):data.index(".dat")], "Zmin_0-CL_1_1-R_0_test_2"),
            test_path + "/" + data.replace(data[data.index("Zmin"):data.index(".dat")], "Zmin_0-CL_1_1-R_0_test_2")
        ]
    for index, (train, test) in enumerate(train_and_test_dict.items(), start=1):
        print(f"processing: {train}, {index}/{len(train_and_test_dict)}")
        SFN = save_file_name + "-" + train.split("/")[-1]
        search_best_params(u_model, u_params, train, test, int(re.search(r"D_(\d+)", train).group(1)), SFN, raw_output,
                           save)


@runtime
def read_json_test_params(file: str) -> dict[str, Any]:
    """
    Read the JSON file which contains the testing parameters

    Parameters:
        file: Path to the JSON file

    Returns:
        dict[str, Any]: Testing parameters
    """
    with open(file, 'r') as f:
        lst = json.load(f)
    return lst
