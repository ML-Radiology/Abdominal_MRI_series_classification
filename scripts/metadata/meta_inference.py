import numpy as np
import pydicom
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, cross_validate, GroupShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, cross_validate, GroupShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt  
from sklearn.metrics import precision_recall_fscore_support as score
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



def calc_feature_importances(model,feat_names,num_to_show):
    # Determine the relative importance of each feature using the random forest model
    importances = model.feature_importances_
    # Get an array of the indices that would sort "importances" in reverse order to get largest to smallest
    indices = np.argsort(importances)[::-1]
    ranked_feats = []
    for i in range(len(indices)):
        feat_name = feat_names[indices[i]]
        ranked_feats.append(feat_name)
    RF_ranking = pd.DataFrame()
    RF_ranking['Feat Index'] = indices
    RF_ranking['Feature'] = ranked_feats
    RF_ranking['Importance'] = np.sort(importances)[::-1]
    #display(RF_ranking.iloc[:num_to_show,:])

    # Plot the importance value for each feature
    RF_ranking[:num_to_show][::-1].plot(x='Feature',y='Importance',kind='barh',figsize=(12,7),legend=False,title='RF Feature Importance')
    plt.show()
    return RF_ranking

# this version is for the dataset with labels and can assess accuracy, etc
def meta_inference(df, scaler, model, feature_list=feats_to_keep):
    # get the features from the preprocessed dataframe (which is the first entity sent by preprocess)
    X = (preprocess(df, scaler=scaler)[0])[feature_list]
    y = df.label
    preds = model.predict(X)
    probs = model.predict_proba(X)
    acc = np.sum(preds==y)/len(y)


    return preds, probs, y, acc


# to get inference on a row of the dataframe that does not necessarily have labels, but which has already been preprocessed
def get_meta_inference(row, scaler, model, features=feats_to_keep): #model_list, feature_list=feats_to_keep):
    # try: 
    #         check_is_fitted(scaler)
    #         # print('This scaler is alrady fitted.')
    # except NotFittedError:
    #         print('This scaler is not fitted.')

    X = (row[features]).values.reshape(1,-1)
    
    pred = model.predict(X)
    probs = model.predict_proba(X)

    return pred, probs

def load_meta_model(model_path):
    with open(model_path, 'rb') as file:
        meta_model = pickle.load(file)
    return meta_model