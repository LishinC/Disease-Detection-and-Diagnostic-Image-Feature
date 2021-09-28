# Revealing unforeseen diagnostic image features with deep learning by detecting cardiovascular diseases from apical four-chamber ultrasounds

- [The why](#The-why)
- [Image features associated with the diagnostic tasks](#Image-features-associated-with-the-diagnostic-tasks)
  - [Representative normal cases](#Representative-normal-cases)
  - [Representative disease cases](#Representative-disease-cases)
- [Loading the trained model weights](#Loading-the-trained-model-weights)
- [Run the code on your own dataset](#Run-the-code-on-your-own-dataset)

In this project, we aimed to develop a deep learning (DL) method to automatically detect impaired left ventricular (LV) function and aortic valve (AV) regurgitation from apical four-chamber (A4C) ultrasound cineloops. Two R(2+1)D convolutional neural networks (CNNs) were trained to detect the respective diseases. Subsequently, tSNE was used to visualize the embedding of the extracted feature vectors, and DeepLIFT was used to identify important image features associated with the diagnostic tasks. 


## The why
* An automated echocardiography interpretation method requiring only limited views as input, say A4C, could make cardiovascular disease diagnosis more accessible.
  * Such system could become beneficial in geographic regions with limited access to expert cardiologists and sonographers. 
  * It could also support general practitioners in the management of patients with suspected CVD, facilitating timely diagnosis and treatment of patients. 

* If the trained CNN can detect the diseases based on limited information, how?
  * Especially, AV regurgitation is typically diagnosed based on color Doppler images using one or more viewpoints. When given only the A4C view, would the model be able to detect regurgitation? If so, what image features does the model use to make the distinction? Since it’s on the A4C view, would the model identify some anatomical structure or movement associated with regurgitation, which are typically not being considered in conventional image interpretation? This is what we try to find out in the study.


## Image features associated with the diagnostic tasks
DeepLIFT attributes a model’s classification output to certain input features (pixels), which allows us to understand which region or frame in an ultrasound is the key that makes the model classify it as a certain diagnosis. Below are some example analyses.

#### Representative normal cases

Case | Averaged logit | Input clip  /  Impaired LV function model's focus  /  AV regurgitation model's focus
----|----|----
Normal1 | 0.9999 | ![image](projectDDDIF/model/DeepLIFT_confident/normal_cases/Normal1_logit0.9999.gif)
Normal2 | 0.9999 | ![image](projectDDDIF/model/DeepLIFT_confident/normal_cases/Normal2_logit0.9999.gif)
Normal3 | 0.9999 | ![image](projectDDDIF/model/DeepLIFT_confident/normal_cases/Normal3_logit0.9999.gif)
Normal4 | 0.9999 | ![image](projectDDDIF/model/DeepLIFT_confident/normal_cases/Normal4_logit0.9999.gif)
Normal5 | 0.9999 | ![image](projectDDDIF/model/DeepLIFT_confident/normal_cases/Normal5_logit0.9999.gif)
Normal6 | 0.9999 | ![image](projectDDDIF/model/DeepLIFT_confident/normal_cases/Normal6_logit0.9999.gif)
Normal7 | 0.9998 | ![image](projectDDDIF/model/DeepLIFT_confident/normal_cases/Normal7_logit0.9998.gif)
Normal8 | 0.9998 | ![image](projectDDDIF/model/DeepLIFT_confident/normal_cases/Normal8_logit0.9998.gif)
Normal9 | 0.9998 | ![image](projectDDDIF/model/DeepLIFT_confident/normal_cases/Normal9_logit0.9998.gif)
Normal10 | 0.9997 | ![image](projectDDDIF/model/DeepLIFT_confident/normal_cases/Normal10_logit0.9997.gif)

DeepLIFT analyses reveal that the LV myocardium and mitral valve were important for detecting impaired LV function, while the tip of the mitral valve anterior leaflet, during opening, was considered important for detecting AV regurgitation.
Apart from the above examples, all confident cases are provided, which the predicted probability of being the normal class by the two models are both higher than 0.98. See the full list [here](projectDDDIF/model/DeepLIFT_confident/normal_cases).

#### Representative disease cases
* Mildly impaired LV

Case | Logit | Input clip  /  Impaired LV function model's focus
----|----|----
MildILV1 | 0.9989 | ![image](projectDDDIF/model/DeepLIFT_confident/disease_cases/MildILV1_logit0.9989.gif)
MildILV2 | 0.9988 | ![image](projectDDDIF/model/DeepLIFT_confident/disease_cases/MildILV2_logit0.9988.gif)

* Severely impaired LV

Case | Logit | Input clip  /  Impaired LV function model's focus
----|----|----
SevereILV1 | 1.0000 | ![image](projectDDDIF/model/DeepLIFT_confident/disease_cases/SevereILV1_logit1.0000.gif)
SevereILV2 | 1.0000 | ![image](projectDDDIF/model/DeepLIFT_confident/disease_cases/SevereILV2_logit1.0000.gif)

* Mild AV regurgitation

Case | Logit | Input clip  /  AV regurgitation model's focus
----|----|----
MildAVR1 | 0.7240 | ![image](projectDDDIF/model/DeepLIFT_confident/disease_cases/MildAVR1_logit0.7240.gif)
MildAVR2 | 0.6893 | ![image](projectDDDIF/model/DeepLIFT_confident/disease_cases/MildAVR2_logit0.6893.gif)

* Substantial AV regurgitation

Case | Logit | Input clip  /  AV regurgitation model's focus
----|----|----
SubstantialAVR1 | 0.9919 | ![image](projectDDDIF/model/DeepLIFT_confident/disease_cases/SubstantialAVR1_logit0.9919.gif)
SubstantialAVR2 | 0.9645 | ![image](projectDDDIF/model/DeepLIFT_confident/disease_cases/SubstantialAVR2_logit0.9645.gif)

When analyzing disease cases, the highlighted regions in different queries are quite different. We speculate that this might be due to a higher heterogeneity in the appearance of the disease cases.
Apart from the above examples, more confident disease cases are provided. See the full list [here](projectDDDIF/model/DeepLIFT_confident/disease_cases).


## Loading the trained model weights
The model weights are made available for external validation, or as pretraining for other echocardiography-related tasks. To load the weights, simply follow:
```
import torch
import torch.nn as nn
import torchvision

#Load impaired LV model
model_path = 'model/impairedLV/train/model_val_min.pth'
# #Load AV regurgitation model
# model_path = 'model/regurg/train/model_val_min.pth'

model = torchvision.models.video.__dict__["r2plus1d_18"](pretrained=False)
model.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(model_path))
```

## Run the code on your own dataset
The dataloader in `util` can be modified to fit your own dataset. To run the full workflow, namely training, validation, testing, and the subsequent analyses, simply run:
```
cd util
pip install -e .
cd ../projectDDDIF
python main.py
```