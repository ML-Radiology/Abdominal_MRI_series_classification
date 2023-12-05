import numpy as np
import pandas as pd
import os
import torch
import pickle
import pydicom
from pydicom.errors   import InvalidDicomError
from pathlib import Path

from fusion_model.fus_model import FusionModel
from cnn.cnn_inference import pixel_inference, load_pixel_model
from metadata.meta_inference import get_meta_inference
from NLP.NLP_inference import get_NLP_inference, load_NLP_model
from config import feats_to_keep, classes, model_paths
from model_container import ModelContainer
from utils import *



# Load the models and create an instance of the ModelContainer
model_container_instance = ModelContainer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_fusion_inference(row, model_container, classes=classes, features=feats_to_keep, device=device, include_nlp=True, use_heuristic=False, conf_threshold=0.7):
    # unpack the models
    metadata_model = model_container.metadata_model
    cnn_model = model_container.cnn_model
    nlp_model = model_container.nlp_model
    scaler = model_container.metadata_scaler

    
    # get metadata preds,probs
    pred1, prob1 = get_meta_inference(row, scaler, metadata_model, features)
    prob1_tensor = torch.tensor(prob1, dtype=torch.float32).squeeze()
    print(pred1)

    # get cnn preds, probs
    pred2, prob2 = pixel_inference(cnn_model, row['fname'].values.tolist()[0], classes=classes)
    prob2_tensor = torch.tensor(prob2, dtype=torch.float32)
    print(pred2)


    if use_heuristic:
        pred3 = None
        prob3 = None
        
        if torch.max(prob1_tensor) > conf_threshold:
            # Use metadata prediction
            predicted_class_idx = torch.argmax(prob1_tensor).item()
            predicted_class = classes[predicted_class_idx]
            confidence_score = torch.max(prob1_tensor).item()
            
        else:
            # Use pixel-based prediction
            predicted_class_idx = torch.argmax(prob2_tensor).item()
            predicted_class = classes[predicted_class_idx]
            confidence_score = torch.max(prob2_tensor).item()
           

    else:
        # Create FusionModel instance
        fusion_model = FusionModel(model_container=model_container, num_classes=len(classes), include_nlp=include_nlp)
        # Load the weights
        if include_nlp:
            weights_path = model_container.fusion_weights_path
        else:
            weights_path = model_container.partial_fusion_weights_path
        
        fusion_model.load_weights(weights_path)
        # get nlp preds, probs...if statement because thinking about assessing both ways
        
        if include_nlp:
            pred3, prob3 = get_NLP_inference(nlp_model, row['fname'].values.tolist()[0], device, classes=classes)
            prob3_tensor = torch.tensor(prob3, dtype=torch.float32)
            print(pred3)
            fused_output = fusion_model(prob1_tensor, prob2_tensor, prob3_tensor)
        else:
            fused_output = fusion_model(prob1_tensor, prob2_tensor)
            pred3 = None
            prob3 = None
        predicted_class = classes[torch.argmax(fused_output, dim=1).item()]
        confidence_score = torch.max(torch.softmax(fused_output, dim=1)).item()

    submodel_df = pd.DataFrame.from_dict({'meta_preds': pred1, 'meta_probs': prob1, 'pixel_preds': pred2, 'pixel_probs': prob2, 'nlp_preds': pred3, 'nlp_probs': prob3, 'SeriesD': row.SeriesDescription})

    return predicted_class, confidence_score, submodel_df

# def get_fusion_inference(self, row, classes=classes, features=feats_to_keep, device=device, include_nlp=True):
def get_fusion_inference_from_file(file_path, model_container, classes=classes, features=feats_to_keep, device=device, include_nlp=True, use_heuristic=False, conf_threshold = 0.7):
    # unpack the models
    metadata_model = model_container.metadata_model
    cnn_model = model_container.cnn_model
    nlp_model = model_container.nlp_model
    scaler = model_container.metadata_scaler
   
   # Create FusionModel instance
    fusion_model = FusionModel(model_container=model_container, num_classes=19)
    # Load the weights
    fusion_model.load_weights(model_container.fusion_weights_path)

    
    my_df = pd.DataFrame.from_dicoms([file_path])

    # Preprocess the metadata using the preprocess function
    preprocessed_metadata, _ = preprocess(my_df, scaler=model_container.metadata_scaler)
    
    # Get the preprocessed row
    row = preprocessed_metadata.iloc[0]

    # Call the original get_fusion_inference function
    predicted_class, confidence_score, troubleshoot_df = get_fusion_inference(row, model_container, classes, features, device, include_nlp, use_heuristic, conf_threshold)

    return predicted_class, confidence_score, troubleshoot_df



def load_fusion_model(model_path):
    with open(model_path, 'rb') as file:
        fusion_model = pickle.load(file)
    return fusion_model