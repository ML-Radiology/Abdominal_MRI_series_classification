
## Abstract:
Accurate, automated MRI series identification is important for many applications, including display (“hanging”) protocols, machine learning, and radiomics. The use of the series description or a pixel-based classifier each has limitations. We demonstrate a combined approach utilizing a DICOM metadata-based classifier and selective use of a pixel-based classifier to identify abdominal MRI series. The metadata classifier was assessed alone as Group metadata and combined with selective use of the pixel-based classifier for predictions with less than 70 % certainty (Group combined). The overall accuracy (mean and 95% confidence intervals) for Groups metadata and combined on the test dataset were 0.870 CI (0.824,0.912) and 0.930 CI (0.893,0.963), respectively.  With this combined metadata and pixel-based approach, we demonstrate accurate classification of 95% or greater for all pre-contrast MRI series and improved performance for some post-contrast series. 

## Dataset:
The dataset is identical to that reported in [2] and is comprised of scans from multiple vendors and field strength scanners at a single institution. It is representative of typical MRI series from clinical abdominal MRI examinations. For each subject there is a single examination, which is typically comprised of 10-15 different series, and in each series there may be a few to more than 100 images of the same type. For series in which more than one set of parameters may be present (such as series containing diffusion weighted imaging with two b-values, or combined dynamic post-contrast series with multiple phases), the subgroups will be separated into distinct series to classify them separately. The original dataset contains 2,215 MRI series for 105 subjects with each subject having a single examination. The dataset was annotated for the series labels by three radiologists with 2-15 years of experience in abdominal imaging.  Nonstandard MRI series used in research protocols and series types with less than 10 examples have been excluded, leaving 19 classes; the training and testing datasets will be randomly selected from the remaining 2165 remaining series with an 80/20 split at the subject-level resulting in 1733 and 432 series, respectively, each with a single label for the series type. 

## Methods and Results:
Custom code was written in the Python programming language (PSF, version 3.8.8) for data processing and analysis. The “pydicom” library (version 2.1.2) was used for DICOM metadata extraction, with several DICOM attributes selected as potential features for machine learning according to the method described by Gauriau et al. [1]. The attributes were then tokenized using one-hot encoding. For DICOM attributes that consisted of a list of strings, each value was converted to a binary feature (presence or absence of the attribute). Attributes containing numerical values were normalized.	The “scikit-learn” library version 0.24.1 was used to train the Random Forest classifier. Grid search was employed for hyperparameter optimization. 


## Metadata Classifier
The metadata classifier is a RandomForest model. A grid search is used to tune hyperparameters, and the model is trained on the resultant optimized model. This can be quickly trained on a cpu, and has fairly high accruacy for many of the types of images. Because the pre and post contrast T1 fat saturation images have identical imaging parameters, the model does not do well in classifying these types of images.


![img.tif](/assets/FigCM_meta02230406.tif)


## Pixel-based Classifier
A single image from each series was selected to create the dataset similar to the method as described by Cluceru et al. [3]. In particular, the middle image from each series was used to represent the entire series for training, validation, and inference. The pixel values were extracted from the DICOM images using “pydicom” and image transformations were applied to the normalized pixel values to produce tensors of the appropriate size and shape for the model (3 x 299 x 299). Data augmentation steps during training included resize, random center crop, and color jitter. The model utilized transfer learning with a DenseNet121 base model trained with ImageNet weights, with the classification layer replaced by a single fully connected layer for the appropriate number of classes (19 classes).   A grid search over optimizers and loss functions led to choosing Adam optimizer and a custom version of Focal Loss tailored for classification with multiple classes. 


Results from current model:
![img.png](/assets/figures/FigCM_dense7020230923.tif)

## NLP Classifier
A stentence transformer was trained over the 'Series Description' field.

## Heuristic Model
The metadata and pixel-based classifiers were used for predictions according to the following rule: if the confidence for the prediction using the metadata classifier was 0.7 or greater, the metadata prediction was used as the overall prediction; if the confidence was lower, the prediction from the pixel-based classifier was used. The accuracy was 93% over the test dataset. 

![img.png](/assets/figures/FigCM_combined.png)

## Fusion Model (FusionModel class)
This is the fully connected layer that takes the concatenated probability vectors from 2 (metadata+pixel modesl) or 3 (also nlp) models. The accuracy is typically below that of the NLP by itself (93% compared with 96%), but higher than that of the pixel and metadata models if all 3 models are used (generally 87-90%, compared with 87-88% for the individual pixel and meta models). The confusion matrix for the fusion model when all 3 submodels are used is shown below. 

![img.png](/assets/figures/FigFusionAll20230416.png)

## DICOM embedding of predictions

One potential method for storing the series predictions to avoid having to get inference of the class every time is to store the prediction in a custom DICOM tag. This is performed in scripts/process_tree.py and is shown in the demo for those cases that have already been processed. The new DICOM field label is PredictedClassInfo and can be found at DICOM metadata locations (0x0011, 0x1010) through (0x0011, 0x1012). In practice, this could be employed on a pass-through server that runs the classifier and then adds the predicted labels to the DICOM metadata, which could then be used in a hanging protocol.

## How to install and use the repository code
**1. Clone this repository**
```
git clone new_name_to_fill_in
```
**2. Install requirements:**
```
pip install -r requirements.txt
```
**3. Change directory to the demo and run the application**
```
cd app
streamlit run demo.py
```
The streamlit demo app will allow one to view images from the sample batch of studies in the source folder in the left sidebar. These images may or may not have labels embedded into the DICOM tags from prior label processing (generally, the prediction will show over the top left aspect of the image if it has been previously processed). One use of the demo app is to select studies to process (one study/patient at a time). This will generate predictions and write them into the DICOM tags by default. If the destination folder selctor is left blank, the default is for the images to be written back to the same folder, overwriting the previously unprocessed study. Other functions in the demo include the ability to get predictions for a single image (after selecting either the heuristic model combining metadata and pixel-based models as described above versus a fusion model). It is also possible to view a study by the series labels (part of the study in the SeriesDescription), or by the predicted class if the study has been previously processed by the classifier. Overall, the goal is to have a pass-through DICOM server that performs the predictions and sends the processed images back to the souce, but this current demo shows proof of concept and provides a user interface to interact with a study of choice. 

**4. Script process_tree.py**

This is what is called by the demo app to process the studies, but could also be called from the command line by
```
cd scripts
python process_tree.py
```
This provides the user more control and allows for processing of an entire directory of studies, and can set behavior like whether previously processed studies should be re-processed (or skipped), and if the desire is to write over previous tags if they are present. 

## Repository Structure
```
.
├── Dockerfile
├── README.md
├── app
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   ├── demo.cpython-311.pyc
│   │   └── demo_utils.cpython-38.pyc
│   ├── demo.py
│   └── demo_utils.py
├── assets
│   ├── FigCM_con_and_7020230918.tif
│   ├── FigCM_con_and_7020230923.tif
│   ├── FigCM_dense20230828.tif
│   ├── FigCM_dense20230919.tif
│   ├── FigCM_dense20230923.tif
│   ├── FigCM_dense7020230919.tif
│   ├── FigCM_dense7020230923.tif
│   ├── FigCM_dense7020231121.tif
│   ├── FigCM_meta20230406.tif
│   ├── FigCM_meta20230828.tif
│   ├── FigCM_meta20230917.tif
│   ├── FigCM_meta20230923.tif
│   ├── FigPixel20230412.png
│   └── figures
│       ├── FigPixel20230322.png
│       └── FigPixel20230322.tif
├── current_packages.txt
├── data
│   ├── fusion_test.pkl
│   ├── fusion_train.pkl
│   ├── fusion_val.pkl
│   ├── image_data_with_label082221.pkl
│   ├── labels.txt
│   ├── testfiles.csv
│   ├── trainfiles.csv
│   └── valfiles.csv
├── models
│   ├── DenseNet_model.pth
│   ├── fusion_model_weights042223.pth
│   ├── fusion_model_weights042423.pth
│   ├── fusion_model_weights20230828.pth
│   ├── fusion_model_weightsDense20230828.pth
│   ├── fusion_model_weightsDense20230919.pth
│   ├── fusion_model_weights_new.pth
│   ├── fusion_model_weights_no_nlp042223.pth
│   ├── fusion_model_weights_no_nlp042423.pth
│   ├── fusion_model_weights_no_nlp20230828.pth
│   ├── fusion_model_weights_no_nlpDense20230828.pth
│   ├── fusion_model_weights_no_nlpDense20230919.pth
│   ├── fusion_model_weights_no_nlpDense20231121.pth
│   ├── fusion_model_weights_no_nlp_new.pth
│   ├── meta_04152023.skl
│   ├── metadata_scaler.pkl
│   ├── pixel_model_041623.pth
│   └── text_model20230415.st
├── notebooks
│   └── Driver_notebook_for_publication_results.ipynb
├── requirements.txt
├── scripts
│   ├── NLP
│   │   ├── NLP_inference.py
│   │   ├── NLP_training.py
│   │   ├── __intit__.py
│   │   └── __pycache__
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── config.cpython-310.pyc
│   │   ├── config.cpython-311.pyc
│   │   ├── config.cpython-37.pyc
│   │   ├── config.cpython-38.pyc
│   │   ├── config.cpython-39.pyc
│   │   ├── model_container.cpython-311.pyc
│   │   ├── model_container.cpython-38.pyc
│   │   ├── process_tree.cpython-311.pyc
│   │   ├── process_tree.cpython-38.pyc
│   │   ├── process_tree.cpython-39.pyc
│   │   ├── train_meta_model.cpython-38.pyc
│   │   ├── train_pixel_model.cpython-38.pyc
│   │   ├── train_text_model.cpython-38.pyc
│   │   ├── utils.cpython-310.pyc
│   │   ├── utils.cpython-311.pyc
│   │   ├── utils.cpython-37.pyc
│   │   ├── utils.cpython-38.pyc
│   │   └── utils.cpython-39.pyc
│   ├── cnn
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── __init__.cpython-37.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── __init__.cpython-39.pyc
│   │   │   ├── cnn_data_loaders.cpython-311.pyc
│   │   │   ├── cnn_data_loaders.cpython-37.pyc
│   │   │   ├── cnn_data_loaders.cpython-38.pyc
│   │   │   ├── cnn_data_loaders.cpython-39.pyc
│   │   │   ├── cnn_dataset.cpython-311.pyc
│   │   │   ├── cnn_dataset.cpython-37.pyc
│   │   │   ├── cnn_dataset.cpython-38.pyc
│   │   │   ├── cnn_dataset.cpython-39.pyc
│   │   │   ├── cnn_inference.cpython-311.pyc
│   │   │   ├── cnn_inference.cpython-38.pyc
│   │   │   ├── cnn_inference.cpython-39.pyc
│   │   │   ├── cnn_model.cpython-311.pyc
│   │   │   ├── cnn_model.cpython-37.pyc
│   │   │   ├── cnn_model.cpython-38.pyc
│   │   │   └── cnn_model.cpython-39.pyc
│   │   ├── cnn_data_loaders.py
│   │   ├── cnn_dataset.py
│   │   ├── cnn_inference.py
│   │   ├── cnn_model.py
│   │   └── cnn_training.py
│   ├── config.py
│   ├── fusion_model
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── __init__.cpython-39.pyc
│   │   │   ├── fus_inference.cpython-311.pyc
│   │   │   ├── fus_inference.cpython-38.pyc
│   │   │   ├── fus_inference.cpython-39.pyc
│   │   │   ├── fus_model.cpython-311.pyc
│   │   │   ├── fus_model.cpython-38.pyc
│   │   │   ├── fus_model.cpython-39.pyc
│   │   │   ├── fus_training.cpython-311.pyc
│   │   │   └── fus_training.cpython-38.pyc
│   │   ├── fus_inference.py
│   │   ├── fus_model.py
│   │   ├── fus_model_old.py
│   │   └── fus_training.py
│   ├── metadata
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── meta_inference.cpython-311.pyc
│   │   │   ├── meta_inference.cpython-38.pyc
│   │   │   ├── meta_training.cpython-311.pyc
│   │   │   └── meta_training.cpython-38.pyc
│   │   ├── meta_inference.py
│   │   └── meta_training.py
│   ├── model_container.py
│   ├── process_tree.py
│   ├── updated_current_packages.txt
│   └── utils.py
└── tree_structure.txt

17 directories, 133 files

```


## References:
1.	Gauriau R, Bridge C, Chen L, Kitamura F, Tenenholtz NA, Kirsch JE, Andriole KP, Michalski MH, Bizzo BC: Using DICOM Metadata for Radiological Image Series Categorization: a Feasibility Study on Large Clinical Brain MRI Datasets,  Journal of Digital Imaging (2020) 33:747-762.
2.	Zhu Z, Mittendorf A, Shropshire E, Allen B, Miller CM, Bashir MR, Mazurowski MA: 3D Pyramid Pooling Network for Liver MRI Series Classification,   IEEE Trans Pattern Anal Mach Intell. 2020 Oct 28. PMID 33112740.
3.	Cluceru J, Lupo JM, Interian Y, Bove R, Crane JC: Improving the Automatic Classification of Brain MRI Acquisition Contrast with Machine Learning, Journal of Digital Imaging, July 2022.

