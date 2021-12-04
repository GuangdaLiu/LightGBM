import copy

import itertools

import math

import pickle

import platform

import random

from pathlib import Path



import numpy as np

import psutil

import pytest

from scipy.sparse import csr_matrix, isspmatrix_csc, isspmatrix_csr

from sklearn.datasets import load_svmlight_file, make_multilabel_classification

from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from sklearn.model_selection import GroupKFold, TimeSeriesSplit, train_test_split



import lightgbm as lgb



# from .utils import load_boston, load_breast_cancer, load_digits, load_iris

from utils import load_boston, load_breast_cancer, load_digits, load_iris



def test_reset_params_works_with_metric_num_class_and_boosting():

    X, y = load_breast_cancer(return_X_y=True)

    dataset_params = {"max_bin": 150}

    booster_params = {

        'objective': 'multiclass',

        'max_depth': 4,

        'bagging_fraction': 0.8,

        'metric': ['multi_logloss', 'multi_error'],

        'boosting': 'gbdt',

        'num_class': 5

    }

    dtrain = lgb.Dataset(X, y, params=dataset_params)

    bst = lgb.Booster(

        params=booster_params,

        train_set=dtrain

    )



    expected_params = dict(dataset_params, **booster_params)

    assert bst.params == expected_params



    booster_params['bagging_fraction'] += 0.1

    print(booster_params)

    print(bst.params)

    new_bst = bst.reset_parameter(booster_params)



    expected_params = dict(dataset_params, **booster_params)

    assert bst.params == expected_params

    assert new_bst.params == expected_params




test_reset_params_works_with_metric_num_class_and_boosting()