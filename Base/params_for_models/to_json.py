import json
import numpy as np
from typing import Any

from numpy.conftest import dtype


def ridgeclassifier() -> None:
    lst: dict[str, Any] = {
        'alpha': [*np.logspace(-4, 4, 9)],
        'class_weight': 'balanced',
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
    }

    with open('RidgeClassifier_params.json', 'w') as f:
        json.dump(lst, f)


def PassiveAggressiveClassifier() -> None:
    lst: dict[str, Any] = {
        'C': [*np.logspace(-4, 4, 9)],
        'class_weight': 'balanced',
        'loss': ['hinge', 'squared_hinge'],
        'n_jobs': -1
    }

    with open('PassiveAggressiveClassifier_params.json', 'w') as f:
        json.dump(lst, f)


def SGDClassifier() -> None:
    lst: dict[str, Any] = {
        'alpha': [*np.logspace(-4, 4, 9)],
        'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber',
                 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet', 'None'],
        'l1_ratio': [0.0, 1.0, 0.15, 0.5],
        'epsilon': [0.01, 0.05, 0.1, 0.5, 1.0],
        'n_jobs': -1,
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'eta0': [1e-4, 1e-3, 1e-2, 0.1, 1.0],
        'power_t': [0.25, 0.5, 0.75, 1.0],
        'early_stopping': [True, False]
    }

    with open('SGDClassifier_params.json', 'w') as f:
        json.dump(lst, f)


def SVC() -> None:
    lst: dict[str, Any] = {
        'C': [*np.logspace(-4, 4, 9)],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.1, 0.5, 1.0],
        'shrinking': [True, False],
        'probability': [True, False],
        'class_weight': 'balanced'
    }

    with open('SVC_params.json', 'w') as f:
        json.dump(lst, f)


def SVClinear() -> None:
    lst: dict[str, Any] = {
        'C': [*np.logspace(-4, 4, 9)],
        'kernel': 'linear',
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.1, 0.5, 1.0],
        'shrinking': [True, False],
        'probability': [True, False],
    }
    with open('SVClinear_params.json', 'w') as f:
        json.dump(lst, f)


def SVCpoly() -> None:
    lst: dict[str, Any] = {
        'C': [*np.logspace(-4, 4, 9)],
        'kernel': 'poly',
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.1, 0.5, 1.0],
        'shrinking': [True, False],
        'probability': [True, False],
        'class_weight': 'balanced'
    }
    with open('SVCpoly_params.json', 'w') as f:
        json.dump(lst, f)


def SVCrbf() -> None:
    lst: dict[str, Any] = {
        'C': [*np.logspace(-4, 4, 9)],
        'kernel': 'rbf',
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.1, 0.5, 1.0],
        'shrinking': [True, False],
        'probability': [True, False],
    }
    with open('SVCrbf_params.json', 'w') as f:
        json.dump(lst, f)


def SVCsigmoid() -> None:
    lst: dict[str, Any] = {
        'C': [*np.logspace(-4, 4, 9)],
        'kernel': 'sigmoid',
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.1, 0.5, 1.0],
        'shrinking': [True, False],
        'probability': [True, False],
        'class_weight': 'balanced'
    }
    with open('SVCsigmoid_params.json', 'w') as f:
        json.dump(lst, f)


def NuSVC() -> None:
    lst: dict[str, Any] = {
        'nu': [*np.arange(0.0, 1.1, 0.25)],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.1, 0.5, 1.0],
        'shrinking': [True, False],
        'probability': [True, False],
        'class_weight': 'balanced',
        'decision_function_shape': ['ovo', 'ovr']
    }

    with open('NuSVC_params.json', 'w') as f:
        json.dump(lst, f)


def NuSVClinear() -> None:
    lst: dict[str, Any] = {
        'nu': [*np.arange(0.0, 1.1, 0.25)],
        'kernel': 'linear',
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.1, 0.5, 1.0],
        'shrinking': [True, False],
        'probability': [True, False],
        'class_weight': 'balanced',
        'decision_function_shape': ['ovo', 'ovr']
    }
    with open('NuSVClinear_params.json', 'w') as f:
        json.dump(lst, f)


def NuSVCpoly() -> None:
    lst: dict[str, Any] = {
        'nu': [*np.arange(0.0, 1.1, 0.25)],
        'kernel': 'poly',
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.1, 0.5, 1.0],
        'shrinking': [True, False],
        'probability': [True, False],
        'class_weight': 'balanced',
    }
    with open('NuSVCpoly_params.json', 'w') as f:
        json.dump(lst, f)


def NuSVCrbf() -> None:
    lst: dict[str, Any] = {
        'nu': [*np.arange(0.0, 1.1, 0.25)],
        'kernel': 'rbf',
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.1, 0.5, 1.0],
        'shrinking': [True, False],
        'probability': [True, False],
        'class_weight': 'balanced',
    }
    with open('NuSVCrbf_params.json', 'w') as f:
        json.dump(lst, f)


def NuSVCsigmoid() -> None:
    lst: dict[str, Any] = {
        'nu': [*np.arange(0.0, 1.1, 0.25)],
        'kernel': 'sigmoid',
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.1, 0.5, 1.0],
        'shrinking': [True, False],
        'probability': [True, False],
        'class_weight': 'balanced',
    }
    with open('NuSVCsigmoid_params.json', 'w') as f:
        json.dump(lst, f)


def LinearSVC() -> None:
    lst: dict[str, Any] = {
        'penalty': ['l1', 'l2'],
        'loss': ['hinge', 'squared_hinge'],
        'dual': [True, False, 'auto'],
        'fit_intercept': [True, False],
        'class_weight': 'balanced'
    }

    with open('LinearSVC_params.json', 'w') as f:
        json.dump(lst, f)


def SGDClassifier() -> None:
    lst: dict[str, Any] = {
        'alpha': [*np.logspace(-4, 4, 9)],
        'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber',
                 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet', 'None'],
        'l1_ratio': [0.0, 1.0, 0.15, 0.5],
        'epsilon': [0.01, 0.05, 0.1, 0.5, 1.0],
        'n_jobs': -1,
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'fit_intercept': [True, False],
        'eta0': [1e-4, 1e-3, 1e-2, 0.1, 1.0],
        'power_t': [0.25, 0.5, 0.75, 1.0],
        'class_weight': 'balanced'
    }

    with open('SGDClassifier_params.json', 'w') as f:
        json.dump(lst, f)


def KNeighborsClassifier() -> None:
    lst: dict[str, Any] = {
        'n_neighbors': [*range(1, 31)],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [*range(1, 51)],
        'p': [*range(1, 6)],
        'metric': ['minkowski', 'cityblock', 'cosine', 'euclidean', 'haversine', 'l1', 'l2', 'manhattan',
                   'nan_euclidean'],
        'n_jobs': -1
    }
    with open('KNeighborsClassifier_params.json', 'w') as f:
        json.dump(lst, f)


def GaussianProcessClassifier() -> None:
    lst: dict[str, Any] = {
        'n_restarts_optimizer': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'max_iter_predict': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'n_jobs': -1
    }
    with open('GaussianProcessClassifier_params.json', 'w') as f:
        json.dump(lst, f)


def GaussianNB() -> None:
    lst: dict[str, Any] = {
        'var_smoothing': [*np.logspace(-9, 0, 10)]
    }
    with open('GaussianNB_params.json', 'w') as f:
        json.dump(lst, f)


def ComplementNB() -> None:
    lst: dict[str, Any] = {
        'alpha': [*np.logspace(-4, 4, 9)],
        'norm': [True, False],
        'fit_prior': [True, False],
        'force_alpha': [True, False]
    }
    with open('ComplementNB_params.json', 'w') as f:
        json.dump(lst, f)


def DecisionTreeClassifier() -> None:
    lst: dict[str, Any] = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': [*range(1, 11)],
        'min_samples_split': [*range(2, 11)],
        'min_samples_leaf': [*range(1, 11)],
        'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'max_features': ['sqrt', 'log2', 'None'],
        'max_leaf_nodes': [*range(1, 11)],
        'class_weight': 'balanced'
    }

    with open('DecisionTreeClassifier_params.json', 'w') as f:
        json.dump(lst, f)


def ExtraTreeClassifier() -> None:
    lst: dict[str, Any] = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': [*range(1, 11)],
        'min_samples_split': [*range(2, 11)],
        'min_samples_leaf': [*range(1, 11)],
        'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'max_features': ['sqrt', 'log2', 'None'],
        'max_leaf_nodes': [*range(1, 11)],
        'class_weight': 'balanced'
    }

    with open('ExtraTreeClassifier_params.json', 'w') as f:
        json.dump(lst, f)


def HistGradientBoostingClassifier() -> None:
    lst: dict[str, Any] = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'max_iter': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'max_leaf_nodes': [*range(100, 1001, 1)],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_leaf': [*range(1, 101, 1)],
        'l2_regularization': [*range(1, 11, 1)],
        'max_bins': [*range(64, 255, 1)],
        'early_stopping': [True, False],
        'class_weight': 'balanced'
    }
    with open('HistGradientBoostingClassifier_params.json', 'w') as f:
        json.dump(lst, f)


def RandomForestClassifier() -> None:
    lst: dict[str, Any] = {
        'n_estimators': [*range(100, 1001, 1)],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [*range(1, 11), 'None'],
        'min_samples_split': [*range(2, 11)],
        'min_samples_leaf': [*range(1, 11)],
        'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'max_features': ['sqrt', 'log2'],
        'max_leaf_nodes': [*range(1, 11), 'None'],
        'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'bootstrap': [True, False],
        'n_jobs': -1,
        'class_weight': ['balanced', 'balanced_subsample']
    }
    with open('RandomForestClassifier_params.json', 'w') as f:
        json.dump(lst, f)


def main() -> None:
    SVC()
    SVClinear()
    SVCpoly()
    SVCrbf()
    SVCsigmoid()


if __name__ == '__main__':
    main()
