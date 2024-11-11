# Frequency Transformer with Local Feature Enhancement for Vehicle Re-Identification
## ABSTRACT 
With the rapid development and widespread adoption of intelligent security, there is an exponential increase in the demand for vehicle re-identification. Vehicle re-Identification involves recognizing the same vehicle across separate cameras, with emphasis on enhancing the processing of local information. Transformer have shown promising results in vehicle re-identification. However, their
inherent self-attention mechanism, which possesses a global receptive field, tends to dilute high frequency texture details when extracting features, hindering the extraction of local features. Additionally, in the presence of challenges such as occlusion and misalignment, transformer may lead to information loss and the introduction of noise, thereby reducing the accuracy of re-identification. To tackle these challenges, the Frequency Transformer with Local Feature Enhancement (LFFT) is introduced. The proposed framework is equipped with Frequency Layer and a Jigsaw Select Patches Module (JSPM). The former focus on enhancing high-frequency component features at the bottom of model, using Frequency Layers consisting of the Fast Fourier Transform. Meanwhile, it continues to extract global features at the top using Attention Layers. In order to generate more robust features, JSPMconsidersrandomlyshufflingandregroupingpatches,thenincorporatingdiscriminativepatches obtained from Attention Layers into groups. This method does not increase the computational and model complexity. Experimental evaluations conducted on two vehicle re-identification datasets, VeRi-776 and VehicleID, affirm the efficacy of the proposed method compared to recent approaches.
## Requirements
### Environment
```
pip install -r requirements.txt
torch 1.6.0
torchvision 0.7.0
timm 0.3.2
cuda 10.1
```
### DataSets
vehicle datasets [VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html),[VeRi-776](https://github.com/JDAI-CV/VeRidataset), Then unzip them and rename them under the directory like
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

### JSPM
