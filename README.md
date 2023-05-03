# Thesis B
Repository for Thesis B work at Maquarie University. 

## Project Summary 
The goal for this project is to design and develop a scalable and fast anomaly detection method for malware detection in the context of IoT or edge applications. A thorough explanation alongside reasonable logic would be required if we are not able to achieve this.

Currently, the approach utilises a Recurrent Neural Network (RNN) - Long Short Term Memory (LSTM) approach. It is composed of three common parts: 
- Data Preprocessing
    - Done thorugh normalisation of dataset, data sampling and data cleansing 
- Feature Extraction 
    - Done through Principal Component Analysis (PCA) 
- Detection Technique
    - Done through RNN-LSTM implementation 

The approach is currently being used against a benign malware Iot dataset (Mirai) and is producing these results: 
- Loss - 3.4
- Accuracy - 30%
- Val_Loss - 4
- Val_Accuracy - 28%

NOTE
- Subject to change as project progresses 
