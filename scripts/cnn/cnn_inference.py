import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pydicom
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import sys
import os


#local imports
from .cnn_model import CustomResNet50, CustomDenseNet
from .cnn_data_loaders import get_data_loaders, data_transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import classes
from utils import create_datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# this will use the test dataset to assess the pixel-based model
def test_pix_model(model,test_loader,device=device):
    '''
    Assesses the pixel_based CNN model over the test dataset. 
    Input: 
        model(model): the trained model being assessed
        test_loader(data_loader): facilitates loading in the test dataset
        device(cpu or gpu)

    Output:
        test_acc(float): accuracy over the test dataset
        recall_vals(list[floats]): the recall values for each class
    '''
    model = model.to(device)
    # Turn autograd off
    with torch.no_grad():

        # Set the model to evaluation mode
        model.eval()

        # Set up lists to store true and predicted values
        y_true = []
        test_preds = []

        # Calculate the predictions on the test set and add to list
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # Feed inputs through model to get raw scores
            logits = model.forward(inputs)
            # Convert raw scores to probabilities (not necessary since we just care about discrete probs in this case)
            probs = F.softmax(logits,dim=1)
            # Get discrete predictions using argmax
            preds = np.argmax(probs.cpu().numpy(),axis=1)
            # Add predictions and actuals to lists
            test_preds.extend(preds)
            y_true.extend(labels.cpu())

        # Calculate the accuracy
        test_preds = np.array(test_preds)
        y_true = np.array(y_true)
        test_acc = np.sum(test_preds == y_true)/y_true.shape[0]
        
        # Recall for each class
        recall_vals = []
        for i in range(len(classes)):
            class_idx = np.argwhere(y_true==i)
            total = len(class_idx)
            correct = np.sum(test_preds[class_idx]==i)
            recall = correct / total
            recall_vals.append(recall)
    
    return test_acc,recall_vals


# gets a tensor for a single image given by its filename
def image_to_tensor(filepath, transforms = data_transforms, device=device):
    
    ds = pydicom.dcmread(filepath)
    img = np.array(ds.pixel_array, dtype=np.float32)
    img = img[np.newaxis]
    img = torch.from_numpy(np.asarray(img))
    input_tensor = transforms['test'](img)

    # Add a batch dimension to the input tensor
    input_tensor = input_tensor.unsqueeze(0)
    #print('changing input_tensor to shape', input_tensor.shape)
    
    # Move the input tensor to the appropriate device
    input_tensor = input_tensor.to(device)

    return input_tensor

def pixel_inference(model, filelist, classes=classes, device=device):
    model = model.to(device)
    # Turn autograd off
    with torch.no_grad():
        model.eval()

    preds = []
    probs = []
    
    count = 0

    if isinstance(filelist, str):
        filelist = [filelist]
    for file in filelist:
        #print('on item ', count, file)
        
        each_tensor = image_to_tensor(file)
        #visualization of a batch of images
        each_tensor = each_tensor.to(device)
        #print('shape of each_tensor is', each_tensor.shape)
        # Feed inputs through model to get raw scores
        logits = model.forward(each_tensor)
        
        
        prob = torch.softmax(logits, dim=1)
        prob = prob.detach().cpu().numpy()
        #print(prob, prob.shape)
        # Get discrete predictions using argmax
        pred = classes[np.argmax(prob)]
        # Add predictions and actuals to lists
        preds.append(pred)
        probs.append(prob)
          

        count+= 1
    # Convert lists to numpy arrays
    predictions_array = np.array(preds).flatten()
    probabilities_array = np.array(probs).squeeze()

    return predictions_array, probabilities_array


def load_pixel_model(modelpath, device=device, output_units = 19, model_type = 'ResNet50'):
    '''
    Loads the model for the CNN assessment. 
    Input: 
        modelpath(path): path to the model
        device(cpu or gpu)
        outputunits(int): number of classes

    Output:
        model(model): Resnet50 transfer learning model
    '''
    if model_type=='ResNet50':
        model = models.resnet50(pretrained=True) # Load the ResNet50 model 

        # Replace the output layer to match the number of output units in your fine-tuned model
        num_finetuned_output_units = output_units
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_finetuned_output_units)

        # Load the saved state_dict
        state_dict = torch.load(modelpath, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    else:
        model = CustomDenseNet(pretrained=False)
        state_dict = torch.load(modelpath, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict) 

    model=model.to(device)
    return model


# Display a batch of predictions
def visualize_results(model,dataloader, classes=classes, device=device):
    model = model.to(device) # Send model to GPU if available
    with torch.no_grad():
        model.eval()
        # Get a batch of validation images
        images, labels = next(iter(dataloader))
        images, labels = images.to(device), labels.to(device)
        # Get predictions
        _,preds = torch.max(model(images), 1)
        preds = np.squeeze(preds.cpu().numpy())
        images = images.cpu().numpy()

    # Plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(15, 10))
    for idx in np.arange(len(preds)):
        ax = fig.add_subplot(2, len(preds)//2, idx+1, xticks=[], yticks=[])
        image = images[idx]
        image = image.transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (std * image + mean)
        
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx] else "red"))
    return

def main():
    # Create instances of model, criterion, optimizer, and scheduler
    # For example:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'pixel_model_041623.pth')
    model = load_pixel_model(model_path)
    
    # with open('../models/meta_and_pixel_fusion_model041623.pkl', 'rb') as file:
    # fusion_model_part = pickle.load(file)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    

    # Get data loaders
    batch_size = 8
    #train, val, test need to be imported
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
    train_datafile = os.path.join(data_path, 'trainfiles.csv')
    val_datafile = os.path.join(data_path,'valfiles.csv')
    test_datafile = os.path.join(data_path,'testfiles.csv')
    train, val, test = create_datasets(train_datafile, val_datafile, test_datafile)

    # dataloaders
    train_loader, val_loader, test_loader, dataset_sizes = get_data_loaders(train, val, test, batch_size)

    # Perform inference on the test dataset
    test_acc, recall_vals = test_pix_model(model, test_loader, device)
    print("Test accuracy:", test_acc)
    print("Recall values:", recall_vals)

    # Perform inference on a single image
    example = test.iloc[100]
    image_path = example.fname
    pred, probs = pixel_inference(model, image_path)
    print("Prediction for the single image:", pred)
    print('The series description for that item is ', example.SeriesDescription)

if __name__ == "__main__":
    main()