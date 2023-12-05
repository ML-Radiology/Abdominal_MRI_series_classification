###### For configuration of the training or inference of the models ######
import torchvision
from torchvision import datasets, models, transforms
import pandas as pd


  ### locations of assets ###
# file_dict = {
# 'img_data_dir_colab':  '/content/gdrive/MyDrive/WW_MRI_abd2/split/',
# 'img_data_dir_local': '/volumes/cm7/Abdominal_MRI_dataset_split/',
# 'txt_data_dir':  '../data/',
# 'test_datafile': '../data/X_test02282023.pkl',
# 'train_datafile': '../data/X_train02282023.pkl',
# 'dataset_file': './stored_assets/dataset.pkl',
# 'train_csv_file': 'trainfiles.csv',
# 'test_csv_file': 'testfiles.csv',
# 'metadata_model_file':  './stored_assets/metadata_model.pkl',
# 'pixel_model_file': './stored_assets/pixel_model_file.pkl',
# 'series_description_model_file': './stored_assets/series_description_model_file.pkl',
# 'labels_file': '../data/labels.txt'
# }

# model_paths = {'cnn': '../models/pixel_model_041623.pth', 'cnnDense': '../models/DenseNet_model.pth', 'nlp': '../models/text_model20230415.st', 'meta': '../models/meta_04152023.skl', 'fusion': '../models/fusion_model_weightsDense20230919.pth', 'fusion_no_nlp': '../models/fusion_model_weights_no_nlpDense20230919.pth', 'scaler': '../models/metadata_scaler.pkl' }

import os

# Determine the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Locations of assets
file_dict = {
    'img_data_dir_colab': os.path.join(PROJECT_ROOT, 'content', 'gdrive', 'MyDrive', 'WW_MRI_abd2', 'split'),
    'img_data_dir_local': os.path.join(PROJECT_ROOT, 'volumes', 'cm7', 'Abdominal_MRI_dataset_split'),
    'txt_data_dir': os.path.join(PROJECT_ROOT, 'data'),
    'test_datafile': os.path.join(PROJECT_ROOT, 'data', 'X_test02282023.pkl'),
    'train_datafile': os.path.join(PROJECT_ROOT, 'data', 'X_train02282023.pkl'),
    #'dataset_file': os.path.join(PROJECT_ROOT, 'stored_assets', 'dataset.pkl'),
    'train_csv_file': os.path.join(PROJECT_ROOT,'data', 'trainfiles.csv'),
    'test_csv_file': os.path.join(PROJECT_ROOT, 'data', 'testfiles.csv'),
    #'metadata_model_file': os.path.join(PROJECT_ROOT, 'stored_assets', 'metadata_model.pkl'),
    #'pixel_model_file': os.path.join(PROJECT_ROOT, 'stored_assets', 'pixel_model_file.pkl'),
    #'series_description_model_file': os.path.join(PROJECT_ROOT, 'stored_assets', 'series_description_model_file.pkl'),
    'labels_file': os.path.join(PROJECT_ROOT, 'data', 'labels.txt')
}

model_paths = {
    'cnn': os.path.join(PROJECT_ROOT, 'models', 'pixel_model_041623.pth'),
    'cnnDense': os.path.join(PROJECT_ROOT, 'models', 'DenseNet_model.pth'),
    'nlp': os.path.join(PROJECT_ROOT, 'models', 'text_model20230415.st'),
    'meta': os.path.join(PROJECT_ROOT, 'models', 'meta_04152023.skl'),
    'fusion': os.path.join(PROJECT_ROOT, 'models', 'fusion_model_weightsDense20230919.pth'),
    'fusion_no_nlp': os.path.join(PROJECT_ROOT, 'models', 'fusion_model_weights_no_nlpDense20230919.pth'),
    'scaler': os.path.join(PROJECT_ROOT, 'models', 'metadata_scaler.pkl')
}



#validation split
val_list =  [41, 84, 14, 25, 76, 47,62,0,55,63,101,18,81,3,4,95,66] #using same train/val/test split as in the original split based on the metadata classifier
random_seed = 42
train_val_split_percent = 0.2
exclusion_labels = [21,22,26,27,28,29,1]
classes =  [0,  2,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 25]

#text model 
sentence_encoder = 'all-MiniLM-L6-v2'
series_description_column = 'SeriesDescription'

#optimized parameters from grid search 
RF_parameters = {
    'bootstrap': False,
    'ccp_alpha': 0.0,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': 371,
    'max_features': 'auto',
    'max_leaf_nodes': None,
    'max_samples': None,
    'min_impurity_decrease': 0.0,
    'min_impurity_split': None,
    'min_samples_leaf': 2,
    'min_samples_split': 10,
    'min_weight_fraction_leaf': 0.0,
    'n_estimators': 373,
    'n_jobs': 2,
    'oob_score': False,
    'random_state': 0,
    'verbose': 0,
    'warm_start': False}


#metadata feature list
feats = ['MRAcquisitionType', 'AngioFlag', 'SliceThickness', 'RepetitionTime',
       'EchoTime', 'EchoTrainLength', 'PixelSpacing', 'ContrastBolusAgent',
       'InversionTime', 'DiffusionBValue', 'seq_E', 'seq_EP', 'seq_G',
       'seq_GR', 'seq_I', 'seq_IR', 'seq_M', 'seq_P', 'seq_R', 'seq_S',
       'seq_SE', 'var_E', 'var_K', 'var_MP', 'var_MTC', 'var_N', 'var_O',
       'var_OSP', 'var_P', 'var_S', 'var_SK', 'var_SP', 'var_SS', 'var_TOF',
       'opt_1', 'opt_2', 'opt_A', 'opt_ACC_GEMS', 'opt_B', 'opt_C', 'opt_D',
       'opt_E', 'opt_EDR_GEMS', 'opt_EPI_GEMS', 'opt_F', 'opt_FAST_GEMS',
       'opt_FC', 'opt_FC_FREQ_AX_GEMS', 'opt_FC_SLICE_AX_GEMS',
       'opt_FILTERED_GEMS', 'opt_FR_GEMS', 'opt_FS', 'opt_FSA_GEMS',
       'opt_FSI_GEMS', 'opt_FSL_GEMS', 'opt_FSP_GEMS', 'opt_FSS_GEMS', 'opt_G',
       'opt_I', 'opt_IFLOW_GEMS', 'opt_IR', 'opt_IR_GEMS', 'opt_L', 'opt_M',
       'opt_MP_GEMS', 'opt_MT', 'opt_MT_GEMS', 'opt_NPW', 'opt_P', 'opt_PFF',
       'opt_PFP', 'opt_PROP_GEMS', 'opt_R', 'opt_RAMP_IS_GEMS', 'opt_S',
       'opt_SAT1', 'opt_SAT2', 'opt_SAT_GEMS', 'opt_SEQ_GEMS', 'opt_SP',
       'opt_T', 'opt_T2FLAIR_GEMS', 'opt_TRF_GEMS', 'opt_VASCTOF_GEMS',
       'opt_VB_GEMS', 'opt_W', 'opt_X', 'opt__', 'type_ADC', 'type_DIFFUSION', 'type_DERIVED']


column_lists = {
    'keep': [
        'fname',
        'file_info',
        # Patient info
        'patientID',
        'PatientID',
        # Study info
        'StudyInstanceUID',
        'StudyID',
        'exam',
        # Series info
        'SeriesInstanceUID',
        'SeriesNumber',
        'SeriesDescription',
        'AcquisitionNumber',
        'contrast',
        'plane',
        'series_num',
        'series',
        # Image info and features
        'InstanceNumber',
        #'ImageOrientationPatient',
        'ScanningSequence',
        'SequenceVariant',
        'ScanOptions',
        'MRAcquisitionType',
        'AngioFlag',
        'SliceThickness',
        'RepetitionTime',
        'EchoTime',
        'EchoTrainLength',
        'PixelSpacing',
        'ContrastBolusAgent',
        'InversionTime',
        'DiffusionBValue',
        'ImageType',
        # Labels
        'plane',
        'seq_label',
        'contrast'],

    'dummies': [
        'ScanningSequence',
        'SequenceVariant',
        'ScanOptions',
        'ImageType'],

    'd_prefixes': [
        'seq',
        'var',
        'opt',
        'type'],

    'binarize': [
        'MRAcquisitionType',
        'AngioFlag',
        'ContrastBolusAgent',
        'DiffusionBValue'],

    'rescale': [
        'SliceThickness',
        'RepetitionTime',
        'EchoTime',
        'EchoTrainLength',
        'PixelSpacing',
        'InversionTime'],

    'dicom_cols': [
        'PatientID',
        # Study info
        'StudyInstanceUID',
        'StudyID',
        'StudyDescription', # to filter on "MRI BRAIN WITH AND WITHOUT CONTRAST" in some cases
        'Manufacturer',
        'ManufacturerModelName',
        'MagneticFieldStrength',
        # Series info
        'SeriesInstanceUID',
        'SeriesNumber',
        'SeriesDescription', # needed for labeling series
        'SequenceName', # may be used for labeling series
        'BodyPartExamined', # to filter on "HEAD" or "BRAIN"
        'AcquisitionNumber',
        # Image info and features
        'InstanceNumber', # i.e. image number
        'SOPClassUID', # to filter on "MR Image Storage"
        'ImageOrientationPatient', # to calculate slice orientation (e.g. axial, coronal, sagittal)
        'EchoTime',
        'InversionTime',
        'EchoTrainLength',
        'RepetitionTime',
        'TriggerTime',
        'SequenceVariant',
        'ScanOptions',
        'ScanningSequence',
        'MRAcquisitionType',
        'ImageType',
        'PixelSpacing',
        'SliceThickness',
        'PhotometricInterpretation',
        'ContrastBolusAgent',
        'AngioFlag', 
        'DiffusionBValue']}

# Data cropping and normalization, also converts single channel to 3 channel for the model
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        #transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])}

# converts numeric labels to textual descriptors ###
abd_label_dict = {
    '1': {
        'long': 'Anythingelse',
        'short': 'other',
        'plane': 'other',
        'contrast': 'other'
    },
    '2': {
        'long': 'Arterial T1w',
        'short': 'arterial',
        'plane': 'ax',
        'contrast': '1'
    },
    '3': {
        'long': 'Early Arterial T1w',
        'short': 'early_arterial',
        'plane': 'ax',
        'contrast': '1'
    },
    '4': {
        'long': 'Late Arterial T1w',
        'short': 'late_arterial',
        'plane': 'ax',
        'contrast': '1'
    },
    '5': {
        'long': 'Arterial Subtraction',
        'short': 'arterial_sub',
        'plane': 'ax',
        'contrast': '1'
    },
    '6': {
        'long': 'Coronal Late Dynamic T1w',
        'short': 'dynamic_late',
        'plane': 'cor',
        'contrast': '1'
    },
    '7': {
        'long': 'Coronal T2w',
        'short': 't2 cor',
        'plane': 'cor',
        'contrast': '0'
    },
    '8': {
        'long': 'Axial DWI',
        'short': 'dwi',
        'plane': 'ax',
        'contrast': '0'
    },
    '9': {
        'long': 'Axial T2w',
        'short': 't2 ax',
        'plane': 'ax',
        'contrast': '0'
    },
    '10': {
        'long': 'Coronal DWI',
        'short': 'dwi',
        'plane': 'cor',
        'contrast': '0'
    },
    '11': {
        'long': 'Fat Only',
        'short': 'dixon_fat',
        'plane': 'ax',
        'contrast': '0'
    },
    '12': {
        'long': 'Axial Transitional_Hepatocyte T1w',
        'short': 'hepatobiliary ax',
        'plane': 'ax',
        'contrast': '1'
    },
    '13': {
        'long': 'Coronal Transitional_Hepatocyte T1w',
        'short': 'hepatobiliary cor',
        'plane': 'cor',
        'contrast': '1'
    },
    '14': {
        'long': 'Axial In Phase',
        'short': 'in_phase ax',
        'plane': 'ax',
        'contrast': '0'
    },
    '15': {
        'long': 'Coronal In Phase',
        'short': 'in_phase cor',
        'plane': 'cor',
        'contrast': '0'
    },
    '16': {
        'long': 'Axial Late Dyanmic T1w',
        'short': 'dynamic_equilibrium',
        'plane': 'ax',
        'contrast': '1'
    },
    '17': {
        'long': 'Localizers',
        'short': 'loc',
        'plane': 'unknown',
        'contrast': '0'
    },
    '18': {
        'long': 'MRCP',
        'short': 'mrcp',
        'plane': 'cor',
        'contrast': '0'
    },
    '19': {
        'long': 'Axial Opposed Phase',
        'short': 'opposed_phase ax',
        'plane': 'ax',
        'contrast': '0'
    },
    '20': {
        'long': 'Coronal Opposed Phase',
        'short': 'opposed_phase cor',
        'plane': 'cor',
        'contrast': '0'
    },
    '21': {
        'long': 'Proton Density Fat Fraction',
        'short': 'fat_quant',
        'plane': 'ax',
        'contrast': '0'
    },
    '22': {
        'long': 'Water Density Fat Fraction',
        'short': 'water_fat_quant',
        'plane': 'ax',
        'contrast': '0'
    },
    '23': {
        'long': 'Portal Venous T1w',
        'short': 'portal_venous',
        'plane': 'ax',
        'contrast': '1'
    },
    '24': {
        'long': 'Coronal Precontrast Fat Suppressed T1w',
        'short': 't1_fat_sat cor',
        'plane': 'cor',
        'contrast': '0'
    },
    '25': {
        'long': 'Axial Precontrast Fat Suppressed T1w',
        'short': 't1_fat_sat',
        'plane': 'ax',
        'contrast': '0'
    },
    '26': {
        'long': 'R*2',
        'short': 'r_star_2',
        'plane': 'ax',
        'contrast': '0'
    },
    '27': {
        'long': 'Axial Steady State Free Precession',
        'short': 'ssfse',
        'plane': 'ax',
        'contrast': '0'
    },
    '28': {
        'long': 'Coronal Steady State Free Precession',
        'short': 'ssfse',
        'plane': 'cor',
        'contrast': '1'
    },
    '29': {
        'long': 'Venous Subtraction',
        'short': 'venous_sub',
        'plane': 'ax',
        'contrast': '1'
    },
    '0': {
        'long': 'Axial ADC',
        'short': 'adc',
        'plane': 'ax',
        'contrast': '0'
    }, ## no longer in use below
     '30': {
        'long': 'Axial Post Contrast Fat Suppressed T1w',
        'short': 't1_fat_sat ax',
        'plane': 'ax',
        'contrast': '1'
    },
    '31': {
        'long': 'Coronal Post Contrast Fat Suppressed T1w',
        'short': 't1_fat_sat cor',
        'plane': 'cor',
        'contrast': '1'
    },
    '32': {
        'long': 'Post Contrast Fat Suppressed T1w',
        'short': 't1_fat_sat',
        'plane': 'ax/cor',
        'contrast': '1'
    } }

feats_to_keep = ['MRAcquisitionType', 'AngioFlag','SliceThickness','RepetitionTime','EchoTime','EchoTrainLength',
                 'PixelSpacing','ContrastBolusAgent', 'InversionTime','seq_E','seq_EP','seq_G','seq_IR','seq_P',
                 'seq_R','seq_S','seq_SE','var_E','var_K','var_N','var_O','var_OSP','var_P','var_S','var_SK','var_SP',
                 'var_SS','opt_2','opt_A','opt_ACC_GEMS','opt_D','opt_EDR_GEMS','opt_EPI_GEMS','opt_F','opt_FAST_GEMS',
                 'opt_FC','opt_FC_SLICE_AX_GEMS','opt_FILTERED_GEMS','opt_FS','opt_I','opt_MP_GEMS','opt_NPW',
                 'opt_P','opt_PFF','opt_PFP','opt_S','opt_SAT2','opt_SAT_GEMS','opt_SEQ_GEMS','opt_SP','opt_T',
                 'opt_TRF_GEMS','opt_VASCTOF_GEMS','opt_W','opt_X','type_ADC','type_DIFFUSION','type_DERIVED']