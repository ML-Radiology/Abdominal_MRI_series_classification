import numpy as np
import pandas as pd
import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torchvision
import pydicom


from cnn.cnn_inference import pixel_inference, load_pixel_model
from metadata.meta_inference import get_meta_inference
from NLP.NLP_inference import get_NLP_inference, load_NLP_model
from config import feats_to_keep, classes, model_paths
from model_container import ModelContainer

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
class FusionModel(nn.Module):
    '''
    Class fusion model is the main model for inference, incorporating the submodels meta_model (dicom metadata), cnn_model (pixel-based classifier), nlp_model (nlp of text of SeriesDescription)
    It adds the submodels and the saved weights for the fusion model through the ModelContainer class. 
    Input:
        model_container(ModelContainer class): contains the submodels and the saved weights for the fusion model
        pretrained(bool): If pretrained, brings in the weights of the saved model for inference. Set False for training of the model
        num_classes(int): Number of classes, which in this case is 19. This pulls from the config file the classes list and default is the length of the classes list
        features(list[str]): list of the features to use in the metadata model, which by default is the feats_to_keep from config
        classes(list[str]): list of the classes, from config
        include_nlp(bool): Defaults to yes, a way to configure the fusion model to use just the metadata and cnn models without the nlp or all three models
    Output: 
        The model itself, the results from a forward pass of the model, or the outputs of get_fusion_inference which returns the class, the confidence score, and a dataframe row of the 
        predictions based on the submodels
    '''
    def __init__(self, model_container, pretrained=True, num_classes=len(classes), features=feats_to_keep, classes=classes, include_nlp=True):
        super(FusionModel, self).__init__()
        self.classes = classes
        self.num_classes = num_classes
        self.features = features
        self.include_nlp = include_nlp
        self.model_container = model_container
    
        self.num_inputs = num_classes * 3 if self.include_nlp == True else num_classes * 2

        # Define the layers of the FusionModel
        self.fusion_layer = nn.Linear(self.num_inputs, self.num_classes)

        if pretrained:
            if include_nlp:
                weights_path = self.model_container.fusion_weights_path
            else:
                weights_path = self.model_container.partial_fusion_weights_path
            
            self.load_weights(weights_path)

    def load_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path))

    
    def forward(self, x1, x2, x3):
        
        if self.include_nlp and x3 is not None:
            #print(x1.shape, x2.shape, x3.shape)
            x = torch.cat((x1, x2, x3), dim=1)
        else:
            #print(x1.shape, x2.shape)
            x = torch.cat((x1, x2), dim=1)

        x = self.fusion_layer(x)
        return x

    ## The main way to get an inference of the fusion model prediction on a single row of the dataframe. 
    ## Output is the predicted class, the confidence score (probability of the class prediction), and a 
    def get_fusion_inference(self, row, classes=classes, features=feats_to_keep, device=device, include_nlp=True, use_heuristic=False, conf_threshold=0.7):
        # get metadata preds,probs
        pred1, prob1 = get_meta_inference(row, self.model_container.metadata_scaler, self.model_container.metadata_model, features)
        prob1_tensor = torch.tensor(prob1, dtype=torch.float32).squeeze().unsqueeze(0)
        

        # get cnn preds, probs
        pred2, prob2 = pixel_inference(self.model_container.cnn_model, row['fname'], classes=classes)
        prob2_tensor = torch.tensor(prob2, dtype=torch.float32).unsqueeze(0)
        
        # get NLP preds, probs
        pred3, prob3 = get_NLP_inference(self.model_container.nlp_model, row['fname'], device, classes=classes)
        prob3_tensor = torch.tensor(prob3, dtype=torch.float32).unsqueeze(0)


        if use_heuristic:
            
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

            # get nlp preds, probs...if statement because thinking about assessing both ways
            if include_nlp:
                pred3, prob3 = get_NLP_inference(self.model_container.nlp_model, row['fname'], device, classes=classes)
                prob3_tensor = torch.tensor(prob3, dtype=torch.float32).unsqueeze(0)
                fused_output = self.forward(prob1_tensor, prob2_tensor, prob3_tensor)
            
            else:
                fused_output = self.forward(prob1_tensor, prob2_tensor)
                

            predicted_class = classes[torch.argmax(fused_output, dim=1).item()]
            confidence_score = torch.max(torch.softmax(fused_output, dim=1)).item()

        submodel_df = pd.DataFrame({'meta_preds': pred1, 'meta_probs': [prob1], 'pixel_preds': pred2, 'pixel_probs': [prob2], 'nlp_preds': pred3, 'nlp_probs': [prob3], 'SeriesD': row.SeriesDescription})
        

        return predicted_class, confidence_score, submodel_df
