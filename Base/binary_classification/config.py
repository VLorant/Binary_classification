from .search_for_classifier import read_json_test_params, automated_file_select_search

train__path = "SyntImbNoisyDataForClassification/train"
test__path = "SyntImbNoisyDataForClassification/test"


def ComplementNB() -> None:
    from sklearn.naive_bayes import ComplementNB
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/ComplementNB_params.json")
    automated_file_select_search(train__path, test__path, ComplementNB, conf, "ComplementNB", raw_output=False,
                                 save=True)


def DecisionTreeClassifier() -> None:
    from sklearn.tree import DecisionTreeClassifier
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/DecisionTreeClassifier_params.json")
    automated_file_select_search(train__path, test__path, DecisionTreeClassifier, conf, "DecisionTreeClassifier",
                                 raw_output=False, save=True)


def ExtraTreeClassifier() -> None:
    from sklearn.tree import ExtraTreeClassifier
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/ExtraTreeClassifier_params.json")
    automated_file_select_search(train__path, test__path, ExtraTreeClassifier, conf, "ExtraTreeClassifier",
                                 False, True)


def GaussianNB() -> None:
    from sklearn.naive_bayes import GaussianNB
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/GaussianNB_params.json")
    automated_file_select_search(train__path, test__path, GaussianNB, conf, "GaussianNB",
                                 raw_output=False, save=True)


def GaussianProcessClassifier() -> None:
    from sklearn.gaussian_process import GaussianProcessClassifier
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/GaussianProcessClassifier_params.json")
    automated_file_select_search(train__path, test__path, GaussianProcessClassifier, conf,
                                 "GaussianProcessClassifier", False, True)


def HistGradientBoostingClassifier() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    conf: dict[str, str | list[str]] = read_json_test_params(
        "params_for_models/HistGradientBoostingClassifier_params.json")
    automated_file_select_search(train__path, test__path, HistGradientBoostingClassifier, conf,
                                 "HistGradientBoostingClassifier", raw_output=False, save=True)


def KNeighborsClassifier() -> None:
    from sklearn.neighbors import KNeighborsClassifier
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/KNeighborsClassifier_params.json")
    automated_file_select_search(train__path, test__path, KNeighborsClassifier, conf, "KNeighborsClassifier",
                                 raw_output=False, save=True)


def NuSVC() -> None:
    from sklearn.svm import NuSVC
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/NuSVC_params.json")
    automated_file_select_search(train__path, test__path, NuSVC, conf, "NuSVC", raw_output=False, save=True)


def NuSVClinear() -> None:
    from sklearn.svm import NuSVC
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/NuSVClinear_params.json")
    automated_file_select_search(train__path, test__path, NuSVC, conf, "NuSVClinear", raw_output=False, save=True)


def NuSVCpoly() -> None:
    from sklearn.svm import NuSVC
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/NuSVCpoly_params.json")
    automated_file_select_search(train__path, test__path, NuSVC, conf, "NuSVCpoly", raw_output=False, save=True)


def NuSVCrbf() -> None:
    from sklearn.svm import NuSVC
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/NuSVCrbf_params.json")
    automated_file_select_search(train__path, test__path, NuSVC, conf, "NuSVCrbf", raw_output=False, save=True)


def NuSVCsigmoid() -> None:
    from sklearn.svm import NuSVC
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/NuSVCsigmoid_params.json")
    automated_file_select_search(train__path, test__path, NuSVC, conf, "NuSVCsigmoid", raw_output=False, save=True)


def PassiveAggressiveClassifier() -> None:
    from sklearn.linear_model import PassiveAggressiveClassifier
    conf: dict[str, str | list[str]] = read_json_test_params(
        "params_for_models/PassiveAggressiveClassifier_params.json")
    automated_file_select_search(train__path, test__path, PassiveAggressiveClassifier, conf,
                                 "PassiveAggressiveClassifier", raw_output=False, save=True)


def RandomForestClassifier() -> None:
    from sklearn.ensemble import RandomForestClassifier
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/RandomForestClassifier_params.json")
    automated_file_select_search(train__path, test__path, RandomForestClassifier, conf, "RandomForestClassifier",
                                 raw_output=False, save=True)


def RidgeClassifier() -> None:
    from sklearn.linear_model import RidgeClassifier
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/RidgeClassifier_params.json")
    automated_file_select_search(train__path, test__path, RidgeClassifier, conf, "RidgeClassifier", raw_output=False,
                                 save=True)


def SGDClassifier() -> None:
    from sklearn.linear_model import SGDClassifier
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/SGDClassifier_params.json")
    automated_file_select_search(train__path, test__path, SGDClassifier, conf, "SGDClassifier", False, True)


def SVC() -> None:
    from sklearn.svm import SVC
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/SVC_params.json")
    automated_file_select_search(train__path, test__path, SVC, conf, "SVC", raw_output=False, save=True)


def SVClinear() -> None:
    from sklearn.svm import SVC
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/SVClinear_params.json")
    automated_file_select_search(train__path, test__path, SVC, conf, "SVClinear", raw_output=False, save=True)


def SVCpoly() -> None:
    from sklearn.svm import SVC
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/SVCpoly_params.json")
    automated_file_select_search(train__path, test__path, SVC, conf, "SVCpoly", raw_output=False, save=True)


def SVCrbf() -> None:
    from sklearn.svm import SVC
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/SVCrbf_params.json")
    automated_file_select_search(train__path, test__path, SVC, conf, "SVCrbf", raw_output=False, save=True)


def SVCsigmoid() -> None:
    from sklearn.svm import SVC
    conf: dict[str, str | list[str]] = read_json_test_params("params_for_models/SVCsigmoid_params.json")
    automated_file_select_search(train__path, test__path, SVC, conf, "SVCsigmoid", raw_output=False, save=True)
