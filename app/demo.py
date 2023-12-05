import pydicom
import os, re
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
import glob
import numpy as np
from PIL import Image, ImageDraw
from glob import glob

from demo_utils import check_prediction_tag, load_dicom_data, apply_window_level, normalize_array, get_single_image_inference
from demo_utils import extract_number_from_filename

import sys, os
#sys.path.append("../scripts/")
# Determine the directory of the current script
script_dir = os.path.dirname(__file__)
# Calculate the absolute path to the project's root directory
project_root = os.path.abspath(os.path.join(script_dir, '..'))
print('project root:', project_root)
# Add the project's root directory to the sys.path
sys.path.append(project_root)

import scripts 
from process_tree import Processor 
from fusion_model.fus_model import FusionModel
from fusion_model.fus_inference import get_fusion_inference_from_file
from config import *
from utils import *
from model_container import ModelContainer

from azure.storage.blob import BlobServiceClient

st.set_page_config(page_title="Abdominal MRI Series Classifier", layout="wide")

st.title("Abdominal MRI Series Classifier")
st.subheader("Metadata and Pixel-Based Model")

# Get instances of models for call to process
model_container = ModelContainer()
fusion_model = FusionModel(model_container=model_container, num_classes=19)

# The place to put processed image data
destination_folder = st.sidebar.text_input("Enter destination folder path:", value="")

# Add a toggle (checkbox) to choose between models
use_heuristic = st.sidebar.checkbox("Use Heuristic Model Instead of Fusion", value=False)

# The place to find the image data
start_folder = st.sidebar.text_input("Enter source folder path:", value="/volumes/cm7/start_folder")

selected_images = None

if start_folder and os.path.exists(start_folder) and os.path.isdir(start_folder):
    folder = st.sidebar.selectbox("Select a source folder:", os.listdir(start_folder), index=0)
    selected_folder = os.path.join(start_folder, folder)

    if os.path.exists(selected_folder) and os.path.isdir(selected_folder):
        dicom_df = load_dicom_data(selected_folder)

        if not dicom_df.empty:
            unique_patients = dicom_df["patient"].drop_duplicates().tolist()
            selected_patient = st.selectbox("Select a patient:", unique_patients, key='patient_selectbox')

            unique_exams = dicom_df[dicom_df["patient"] == selected_patient]["exam"].drop_duplicates().tolist()
            selected_exam = st.selectbox("Select an exam:", unique_exams, key='exam_selectbox')

            unique_series = dicom_df[(dicom_df["patient"] == selected_patient) & (dicom_df["exam"] == selected_exam)]["series"].drop_duplicates().tolist()
            selected_series = st.selectbox("Select a series:", unique_series, key='series_selectbox')

            if not dicom_df.empty:
                has_labels = dicom_df[dicom_df["exam"] == selected_exam]["label"].notnull().any()

                if has_labels:
                    unique_labels = dicom_df[(dicom_df["patient"] == selected_patient) & (dicom_df["exam"] == selected_exam)]["label"].drop_duplicates().tolist()
                    selected_label = st.selectbox("Select images predicted to be of type:", unique_labels)
                else:
                    st.write("The selected exam has no labels available in the DICOM tags.")
                    selected_label = None

            source_selector = st.radio("Select source:", ["Series", "Predicted Type"])

            if source_selector == 'Series': 
                selected_images = dicom_df[
                    (dicom_df["patient"] == selected_patient) &
                    (dicom_df["exam"] == selected_exam) &
                    (dicom_df["series"] == selected_series)]["file_path"].tolist()
            
            elif (source_selector == "Predicted Type") and has_labels:
                selected_images = dicom_df[
                    (dicom_df["patient"] == selected_patient) &
                    (dicom_df["exam"] == selected_exam) &
                    (dicom_df["label"] == selected_label)]["file_path"].tolist()

            st.subheader("Selected Study Images")
            cols = st.columns(4)

            if selected_images:
                selected_images.sort(key=lambda x: extract_number_from_filename(os.path.basename(x)))
                image_idx = st.select_slider("View an image", options=range(len(selected_images)), value=0)

                image_path = selected_images[image_idx]
                if not isinstance(image_path, str):
                    image_path = str(image_path)            

                if os.path.isfile(image_path):
                    print(f"{image_path} is a valid file path.")
                else:
                    print(f"{image_path} is not a valid file path.")
                
                dcm_data = pydicom.dcmread(image_path)
                predicted_type, meta_prediction, cnn_prediction, nlp_prediction = check_prediction_tag(dcm_data)

                window_width = st.sidebar.slider("Window Width", min_value=1, max_value=4096, value=2500, step=1)
                window_center = st.sidebar.slider("Window Level", min_value=-1024, max_value=1024, value=0, step=1)
                
                with st.container():
                    image_file = selected_images[image_idx]
                    try:
                        dcm_data = pydicom.dcmread(image_file)
                        image = dcm_data.pixel_array
                        image = apply_window_level(image, window_center=window_center, window_width=window_width)
                        image = Image.fromarray(normalize_array(image))
                        image = image.convert("L")
                        if predicted_type:
                            draw = ImageDraw.Draw(image)
                            text = f"Predicted Type: {predicted_type}"
                            draw.text((10, 10), text, fill="white")
                        
                            if meta_prediction:
                                textm = f'Metadata prediction: {meta_prediction}'
                                draw.text((15, 50), textm, fill="white")
                            if cnn_prediction:
                                textc = f'Pixel-based CNN prediction: {cnn_prediction}'
                                draw.text((15, 100), textc, fill="white")
                            if nlp_prediction:
                                textn = f'Text-based NLP prediction: {nlp_prediction}'
                                draw.text((15, 150), textn, fill="white")
                        else:
                            draw = ImageDraw.Draw(image)
                            text = f'No prediction yet'
                            draw.text((10, 10), text, fill='white')
                        st.image(image, caption=os.path.basename(image_file), use_column_width=True)
                    
                    except Exception as e:
                        pass
                
            
            else:
                st.write('No type of this predicted class in the exam.')

            process_images = st.sidebar.button("Process Images")
            if process_images:
                if not destination_folder:
                    destination_folder = start_folder
                processor = Processor(selected_folder, destination_folder, fusion_model=fusion_model, overwrite=True, write_labels=True, use_heuristic = use_heuristic)

                new_processed_df = processor.pipeline_new_studies()
        
            get_inference = st.button("Get Inference For This Image")
            if get_inference:
                predicted_type, predicted_confidence, prediction_meta, meta_confidence, cnn_prediction, cnn_confidence, nlp_prediction, nlp_confidence = get_single_image_inference(image_path, model_container, fusion_model, use_heuristic = use_heuristic, conf_threshold=0.7)
                st.write(f'Predicted type: {predicted_type}, confidence score: {predicted_confidence:.2f}')
                st.write(f'Metadata prediction:  {prediction_meta}, {meta_confidence:.2f}')
                st.write(f'Pixel CNN prediction: {cnn_prediction}, {cnn_confidence:.2f}')
                st.write(f'Text-based prediction: {nlp_prediction}, {nlp_confidence:.2f}')
        else:
            st.warning("No DICOM files found in the folder.")
else:
    st.error("Invalid start folder path.")
