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
import pydicom

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

def load_NLP_model(nlp_path):
    with open(nlp_path, 'rb') as file:
        NLP_model = pickle.load(file)

    return NLP_model

def get_NLP_inference(model, filenames, device=device, classes=classes):
    
    senttrans_model = SentenceTransformer(sentence_encoder, device=device)
    preds = []
    probs = []

    if isinstance(filenames, str):
        filenames = [filenames]

    for filename in filenames:
        print(filename)
        try:
            ds = pydicom.dcmread(filename)
            description = ds.SeriesDescription
            
            description_encoded = senttrans_model.encode(description)

            #print(f'Getting prediction for file {filename} with SeriesDesription label {description} and shape {description_encoded.shape}')

            pred = model.predict(description_encoded.reshape(1, -1))[0]  # Use description_encoded and reshape to a 2D array
            preds.append(pred)
            prob = model.predict_proba(description_encoded.reshape(1, -1))  # Use description_encoded and reshape to a 2D array
            probs.append(prob)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    preds_np = np.array(preds)  # Convert preds to a NumPy array
    probs_np = np.array(probs).squeeze()  # Convert probs to a NumPy array


    return preds_np, probs_np