import numpy as np
import pydicom
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, cross_validate, GroupShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score,  ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, cross_validate, GroupShuffleSplit, RandomizedSearchCV, GridSearchCV

import matplotlib.pyplot as plt  
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, ConfusionMatrixDisplay
from pprint import pprint
import pickle

try:
    from config import file_dict, feats, feats_to_keep, column_lists
    from config import abd_label_dict, val_list, train_val_split_percent, random_seed, data_transforms
    from config import sentence_encoder, series_description_column
    from config import RF_parameters
    from utils import *
except ImportError:
    from ..config import file_dict, feats, feats_to_keep, column_lists
    from ..config import abd_label_dict, val_list, train_val_split_percent, random_seed, data_transforms
    from ..config import sentence_encoder, series_description_column
    from ..config import RF_parameters
    from ..utils import *


#randomized grid search for hyperparameters
def train_fit_parameter_trial(train, y, features, fname='../models/model-run.skl'):
    #Train a Random Forest classifier on `train[features]` and `y`, then save to `fname` and return
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(train[features], y)
    print('Parameters currently in use:\n')
    pprint(clf.get_params())


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 20, stop = 500, num = 20)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 660, num = 10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 20]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4, 8]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    
    clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=random_seed, n_jobs=-1)
    clf_random.fit(train[features], y)
    opt_clf = clf_random.best_estimator_
    pprint(clf_random.best_params_)
    
    # store the model for inference
    pickle.dump(opt_clf, open(fname, 'wb'))
    return opt_clf

def train_meta_model(X, y, features, params=RF_parameters, fname = 'trained_meta_model.skl'):
    clf = RandomForestClassifier(params)
    clf.fit(X[features], y)
    print('Parameters currently in use:\n')
    pprint(clf.get_params())

    pickle.dump(clf, open(fname, 'wb'))
    #dump(clf_random, fname)
    return clf


def evaluate_meta_model(model, test_features, y_test):
    test_preds = model.predict(test_features)
    test_acc = np.sum(test_preds==y_test)/len(y_test)
    print('Test set accuracy is {:.3f}'.format(test_acc))
    
    return test_acc
   
