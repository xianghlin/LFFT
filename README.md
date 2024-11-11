# Frequency Transformer with Local Feature Enhancement for Vehicle Re-Identification
## ABSTRACT 
With the rapid development of intelligent security systems, the demand for vehicle re-identification has surged exponentially. Vehicle re-identification involves recognizing the same vehicle across different camera perspectives, necessitating robust local feature processing. While transformers have shown promising results in this field, their inherent self-attention mechanism tends to dilute high-frequency texture details, hindering local feature extraction. Additionally, challenges such as occlusion and misalignment can lead to information loss and noise introduction, reducing re-identification accuracy. To address these issues, we introduce the Frequency Transformer with Local Feature Enhancement (LFFT). The proposed framework comprises a Frequency Layer and a Jigsaw Select Patches Module (JSPM). The Frequency Layer enhances the weights of high-frequency component features using Fast Fourier Transform to improve local feature extraction at the lower layers. Meanwhile, the Attention Layers at the higher layers continue to extract global features. The JSPM incorporates discriminative patches obtained from the Attention Layers into randomly shuffled and reorganized groups, enhancing the global discriminative capability of local features. This method does not increase the computational and model complexity. Experimental evaluations on two vehicle re-identification datasets, VeRi-776 and VehicleID, demonstrate the effectiveness of our method compared to recent approaches.
## Requirements
### Environment
```
torch 1.6.0
torchvision 0.7.0
timm 0.3.2
numpy 1.21.6
cuda 10.1
mmcv 1.5.1
mmengine 0.7.3
```
### DataSets
vehicle datasets [VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html), [VeRi-776](https://github.com/JDAI-CV/VeRidataset), Then unzip them and rename them under the directory like
```
data
├── VehicleID_V1.0
│   └── images ..
└── VeRi
    └── images ..
```
## Methods
### Architecture
![architectue_new](https://github.com/user-attachments/assets/b102ebe5-fb02-42c5-aa70-ebefa43056f6)
### Frequency Layer
<img src="https://github.com/user-attachments/assets/3f5b8b89-e943-4987-9693-33e744d8c5e5" width="500x">

### JSPM
<img src="https://github.com/user-attachments/assets/315c1c3b-6093-4fc8-bc18-d50f1949d6cd" width="500x">
<img src="https://github.com/user-attachments/assets/8a363bc0-4b33-43cc-80a0-85882aeca0a7" width="500x">

