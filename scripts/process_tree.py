import numpy as np
import pandas as pd
import os
from fusion_model.fus_inference import get_fusion_inference
from fusion_model.fus_model import FusionModel
from model_container import ModelContainer
import pydicom
from utils import *
from config import model_paths



class Processor:
    '''
    Class which contains code to process a batch one or many DICOM studies contained within
    a folder. It creates and returns a dataframe containing the DICOM metadata and the 
    predicted class for each series (typically several images of the same type traversing the 
    anatomy within a study, a study typically has several and a patient may have one or more studies), 
    as well as the confidence in the prediction (the probability). In addition to creating the
    dataset, the prediction and confidence values will also be written into the DICOM metadata
    in Unknown tags (0011)(1010) and (0011)(1011) that can be used to select the appropriate series
    for a processed examination.  Because images in a certain series are tyipcally of the same 
    type, this classifies at the level of the series rather than the individual image.

    Input: 
        data_dir(str): path to the studies being processed
        destination_foder(str): path to the folder where the proessed images will be written
        write_labels(bool): by default true; whether to write the labels into the DICOM metadata
            and store the new images
        overwrite(bool): whether to process images that already have information in these tags, 
            typically due to prior processing
        fusion_model(model): the classifier model which incorporates the various subcomponents
            (metadata RandomForest, pixel CNN, and textual description NLP models) in fusion
            model that is a single fully connected layer.
    Output: If pipeline_new_studies is run, it will return the processed dataframe
    
    '''
    def __init__(self, data_dir, destination_folder, write_labels=True, overwrite = False, fusion_model=None, remoteflag = False, destblob = None, destclient = None, use_heuristic = False, conf_threshold = 0.7):
        self.data_dir = data_dir
        self.destination_folder = destination_folder
        self.write_labels = write_labels
        self.fusion_model = fusion_model
        self.troubleshoot_df = None
        self.overwrite = overwrite
        self.remoteflag = remoteflag
        self.destblob = destblob
        self.destclient = destclient
        self.use_heuristic = use_heuristic
        self.confidence_threshold = conf_threshold
    
    # The overall active component which preprocesses the dataframe and calls the cascade
    # of actions to process the folder and its subdirectories which are typically 
    # organized by patient, then by study, then by series. The return is the processed
    # dataframe
    def pipeline_new_studies(self):
        # looks for dicom files and loads them into a dataframe
        _, df = get_dicoms(self.data_dir)
        df1 = df.copy()
        
        ## prepare df prior to evaluation
        df1 = expand_filename(df1, ['blank', 'filename', 'series', 'exam', 'patientID'])
        df1.drop(columns='blank', inplace=True)
        df1['file_info']=df1.fname
        df1['img_num'] = df1.file_info.apply(extract_image_number)
        df1['contrast'] = df1.apply(detect_contrast, axis=1)
        df1['plane'] = df1.apply(compute_plane, axis=1)
        df1['series_num'] = df1.series.apply(lambda x: str(x).split('_')[-1])
        with open(model_paths['scaler'], 'rb') as file:
            scaler = pickle.load(file)

        ## gets the features from the metadata for the RF model
        df1, _ = preprocess(df1, scaler)

        processed_frame = self.process_batch(df1)
        return processed_frame

    # at the top of the cascade, acts on the entire batch by calling function for each patient
    def process_batch(self, df):
        df1 = df.copy()
        batch = df1.groupby('patientID').apply(self.process_patient)
        return batch

    # at the next level of the cascade, acts on a patient by processing each exam
    def process_patient(self, patient_df):
        processed_exams = patient_df.groupby('exam').apply(self.process_exam)
        return processed_exams

    # at the next level of the cascade, acts on an exam by processing each series
    def process_exam(self, exam_df):
        processed_series = exam_df.groupby('series').apply(self.process_series)
        return processed_series

    # since all the images in a series are typically of the same type, performs
    # the classification at the series level. Gets the middle image from each series
    # and performs the classification for the 3 modalities and the fusion model based
    # on the middle image given prior results of the pixel CNN classifier showing good
    # results on this strategy
    def process_series(self, series_df, selection_fraction=0.5):
        # Sort the dataframe by file_info (or another relevant column)
        sorted_series = series_df.sort_values(by='fname')

        # Find the middle image index
        middle_index = int(len(sorted_series) * selection_fraction)

        # Get the middle image
        middle_image = sorted_series.iloc[middle_index]

        # Gets classification from the fusion model
        predicted_series_class, predicted_series_confidence, ts_df = self.fusion_model.get_fusion_inference(middle_image, use_heuristic = self.use_heuristic, conf_threshold = self.confidence_threshold)
        # Writes the predictions into the dataframe
        sorted_series['predicted_class'] = predicted_series_class
        sorted_series['prediction_confidence'] = np.round(predicted_series_confidence, 2)
        ## adding the submodel info
        sorted_series['meta_prediction'], sorted_series['meta_probs'] = ts_df['meta_preds'], ts_df['meta_probs']
        sorted_series['cnn_prediction'], sorted_series['cnn_probs'] = ts_df['pixel_preds'], ts_df['pixel_probs']
        sorted_series['nlp_prediction'], sorted_series['nlp_probs'] = ts_df['nlp_preds'], ts_df['nlp_probs']
        submodel_preds_list = [ts_df['meta_preds'].iloc[0], ts_df['pixel_preds'].iloc[0], ts_df['nlp_preds'].iloc[0]]
        submodel_preds_string = "'".join(map(str, submodel_preds_list))
        ## going to also add the ts_df dataframe which contains the submodel preds/probs
        # ts_df_repeated = pd.concat([ts_df]* len(sorted_series), ignore_index=True)
        # sorted_series = pd.concat([sorted_series, ts_df_repeated], axis=1)
        

        # makes a new folder given by the destination_folder if it does not yet exist
        relative_path = os.path.relpath(series_df.fname.iloc[0], self.data_dir)
        save_path = os.path.join(self.destination_folder, os.path.dirname(relative_path))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # writes labels into the DIOM metadata if write_labels is true 
        if self.write_labels:
            #print('writing new data into', save_path)
            self.write_labels_into_dicom(sorted_series, label_num=predicted_series_class,
                            conf_num=np.round(predicted_series_confidence, 3), substring = submodel_preds_string, path=save_path)

        return sorted_series
    


    def write_labels_into_dicom(self, series_group, label_num, conf_num, substring, path):
        #print('writing labels', label_num)
        for dicom_file in series_group.fname.tolist():
            filename = os.path.basename(dicom_file)
            ds = dcmread(dicom_file, no_pixels=False)

            private_creator_tag = pydicom.tag.Tag(0x0011, 0x0010)
            custom_tag1 = pydicom.tag.Tag(0x0011, 0x1010)
            custom_tag2 = pydicom.tag.Tag(0x0011, 0x1011)
            custom_tag3 = pydicom.tag.Tag(0x0011, 0x1012)

            # Check if private creator and custom tags already exist; if overwrite, then proceed anyways
            if (private_creator_tag not in ds or custom_tag1 not in ds or custom_tag2 not in ds) or self.overwrite:
                # Create and add private creator element
                private_creator = pydicom.DataElement(private_creator_tag, 'LO', 'PredictedClassInfo')
                ds[private_creator_tag] = private_creator

                # Create and add custom private tags
                data_element1 = pydicom.DataElement(custom_tag1, 'IS', str(label_num))
                data_element1.private_creator = 'PredictedClassInfo'
                data_element2 = pydicom.DataElement(custom_tag2, 'DS', str(conf_num))
                data_element2.private_creator = 'PredictedClassInfo'
                data_element3 = pydicom.DataElement(custom_tag3, 'LO', substring)
                data_element3.private_creator = 'PredictedClassInfo'
                ds[custom_tag1] = data_element1
                ds[custom_tag2] = data_element2
                ds[custom_tag3] = data_element3

                modified_file_path = os.path.join(path, filename)
                ds.save_as(modified_file_path)
            else: # no overwrite and tags already exist
                print(f"Custom tags already exist in {dicom_file}, skipping this file.")

            modified_file_path = os.path.join(path, filename)
            ds.save_as(modified_file_path)  


def main():
    old_data_site = '/volumes/cm7/source042323/'
    destination_site = '/volumes/cm7/testing_additions042323/'
    
    # get the models
    model_container = ModelContainer()
    fusion_model = FusionModel(model_container = model_container, num_classes=19)
    
    # instantiate the processor class for action on the DICOM images
    processor = Processor(old_data_site, destination_site, fusion_model=fusion_model, write_labels=True)
    new_processed_df = processor.pipeline_new_studies()
    print(new_processed_df)
    new_processed_df.to_pickle('../data/newly_processed_cases042423.pkl')

if __name__ == "__main__":
    main()