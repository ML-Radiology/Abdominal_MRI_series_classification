import numpy as np
import pandas as pd
import os
import torch
import torch.optim as optim
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from fusion_model.fus_model import FusionModel
from scripts.cnn.cnn_inference import pixel_inference, load_pixel_model
from scripts.metadata.meta_inference import get_meta_inference
from scripts.NLP.NLP_inference import get_NLP_inference, load_NLP_model
from scripts.config import feats_to_keep, classes, model_paths
from scripts.model_container import ModelContainer
from scripts.utils import *




class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x1 = torch.tensor(self.dataframe.iloc[idx]['meta_probs'], dtype=torch.float32)
        x2 = torch.tensor(self.dataframe.iloc[idx]['pixel_probs'], dtype=torch.float32)
        x3 = torch.tensor(self.dataframe.iloc[idx]['nlp_probs'], dtype=torch.float32)
        
        label = torch.tensor(self.dataframe.iloc[idx]['true'], dtype=torch.long)

        return x1, x2, x3, label



def train_fusion_model(model, train_loader, val_loader, test_loader, device, optimizer, loss_fn, num_epochs, include_nlp = True):
    model.to(device)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for x1, x2, x3, labels in train_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            if include_nlp:
                x3 = x3.to(device)
                outputs = model(x1, x2, x3)
                #print(outputs.shape)
            else:
                outputs = model(x1, x2, x3=None)
                #print(outputs.shape)


            optimizer.zero_grad()

            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x1.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
        #     for x1, x2, x3, labels in val_loader:
        #         x1, x2, x3, labels = x1.to(device), x2.to(device), x3.to(device), labels.to(device)

        #         outputs = model(x1, x2, x3)
        #         _, preds = torch.max(outputs, 1)
        #         loss = loss_fn(outputs, labels)

        #         running_loss += loss.item() * x1.size(0)
        #         running_corrects += torch.sum(preds == labels.data)

            for x1, x2, x3, labels in val_loader:
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
                if include_nlp:
                    x3 = x3.to(device)
                    outputs = model(x1, x2, x3)
                
                else:
                    outputs = model(x1, x2, x3=None)
                _, preds = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels)

                running_loss += loss.item() * x1.size(0)
                running_corrects += torch.sum(preds == labels.data)    



            epoch_loss = running_loss / len(val_loader.dataset)
            epoch_acc = running_corrects.double() / len(val_loader.dataset)

            print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            history['val_loss'].append(epoch_loss)
            history['val_acc'].append(epoch_acc)

    return model, history


def main():

    #load in pickled dataframes from the base model training:
    with open('data/fusion_train.pkl', 'rb') as file:
        ftrain = pickle.load(file)

    with open('data/fusion_val.pkl', 'rb') as file:
        fval = pickle.load(file)

    with open('data/fusion_test.pkl', 'rb') as file:
        ftest = pickle.load(file)

    # get model container for the base models
    model_container=ModelContainer()

    #  Instantiate FusionModel
    fusion_model = FusionModel(model_container)
    fusion_model_no_nlp = FusionModel(model_container)

    # Define the loss function and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)
    p_optimizer = ptim.Adam(fusion_model_no_nlp.parameters(), lr=0.001)


    train_dataset = CustomDataset(ftrain)
    val_dataset = CustomDataset(fval)
    test_dataset = CustomDataset(ftest)

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    #model, train_loader, val_loader, device, optimizer, loss_fn, num_epochs, include_nlp = True
    trained_model, training_history_with_nlp = train_fusion_model(fusion_model, train_loader, val_loader, device, optimizer, loss_fn = criterion, num_epochs=30, include_nlp=True)
    trained_model_no_nlp, training_history_without_nlp = train_fusion_model(fusion_model_no_nlp, train_loader, val_loader, device, p_optimizer, loss_fn=criterion, num_epochs=30, include_nlp=False)
    
    print(training_history_with_nlp)

    # save trained model
    model_weights_path = 'models/fusion_model_weights_new.pth'
    torch.save(trained_model.state_dict(), model_weights_path)

    model_weights_path_no_nlp = 'models/fusion_model_weights_no_nlp_new.pth'
    torch.save(trained_model_no_nlp.state_dict(), model_weights_path_no_nlp)

    return trained_model, trained_model_no_nlp

