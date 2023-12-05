from __future__ import print_function
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk, spacy
import sklearn
from sklearn.model_selection import train_test_split
import os
import os.path
import glob

import sys
from random import shuffle
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import time
import copy
from datetime import datetime
import pickle 

try:
    from config import file_dict, feats, column_lists, RF_parameters, classes
    from config import abd_label_dict, val_list, train_val_split_percent, random_seed, data_transforms
    from config import sentence_encoder, series_description_column
    from utils import shorten_df, plot_and_save_cm, prepare_df
except ImportError:
    from ..config import file_dict, feats, column_lists, RF_parameters, classes
    from ..config import abd_label_dict, val_list, train_val_split_percent, random_seed, data_transforms
    from ..config import sentence_encoder, series_description_column
    from ..utils import shorten_df, plot_and_save_cm, prepare_df

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
senttrans_model = SentenceTransformer(sentence_encoder, device=device)


def train_NLP_model(train_data, val_data, test_data, senttrans_model=senttrans_model):
    X_train = train_data['SeriesDescription']
    y_train = train_data['label']
    
    X_val = val_data['SeriesDescription']
    y_val = val_data['label']
    
    X_test = test_data['SeriesDescription']
    y_test = test_data['label']

    
    X_train_encoded = [senttrans_model.encode(doc) for doc in X_train.to_list()]
    X_val_encoded = [senttrans_model.encode(doc) for doc in X_val.to_list()]
    X_test_encoded = [senttrans_model.encode(doc) for doc in X_test.to_list()]

    # Train a classification model using logistic regression classifier
    logreg_model = LogisticRegression(solver='saga')
    logreg_model.fit(X_train_encoded, y_train)
    
    train_preds = logreg_model.predict(X_train_encoded)
    train_probs = logreg_model.predict_proba(X_train_encoded)
    train_acc = sum(train_preds == y_train) / len(y_train)
    print('Accuracy on the training set is {:.3f}'.format(train_acc))

    ## assess on the val set
    #print('size of X_val_encoded is ', len(X_val_encoded))
    #print('size of y_val is ', len(y_val))
    val_preds = logreg_model.predict(X_val_encoded)
    val_probs = logreg_model.predict_proba(X_val_encoded)
    print('size of preds_val is ', len(val_preds))
    val_acc = sum(val_preds == y_val)/ len(y_val)
    print('Accuracy on the val set is {:.3f}'.format(val_acc))
    
    ## display results on test set
    test_preds = logreg_model.predict(X_test_encoded)
    test_probs = logreg_model.predict_proba(X_test_encoded)
    test_acc = sum(test_preds == y_test) / len(y_test)
    ## display results on test set
    print('Accuracy on the test set is {:.3f}'.format(test_acc))


    #export model
    #txt_model_filename = "../models/text_model"+ datetime.now().strftime('%Y%m%d') + ".st"
    #pickle.dump(logreg_model, open(txt_model_filename, 'wb'))

    return train_preds, train_probs, train_acc, val_preds, val_probs, val_acc, test_preds, test_probs, test_acc, logreg_model

def list_incorrect_text_predictions(ytrue, ypreds, series_desc):
    ytrue = ytrue.tolist()
    ytrue_label = [abd_label_dict[str(x)]['short'] for x in ytrue]
    ypreds = ypreds.tolist()
    ypreds_label = [abd_label_dict[str(x)]['short'] for x in ypreds]
    ylist = zip(series_desc, ytrue, ypreds)
    ylist_label = zip(series_desc,ytrue_label, ypreds_label)
    y_incorrect_list = [x for x in ylist if x[1]!=x[2]]
    y_incorrect_list_label = [x for x in ylist_label if x[1]!=x[2]]
    
    return y_incorrect_list, y_incorrect_list_label

