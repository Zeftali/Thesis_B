# Thesis B
Repository for Thesis B work at Maquarie University. 

### Load Data 
Essential imports. 
- Necessary classes from the Keras library 
- Includes EarlyStopping callback which is used to stop training when the validation loss stops improving

### Define Model
Definition of layers 
- Currently three LSTM layers, each with a Dropout layer to prevent overfitting 
- A single output layer
- return_sequences = True for first two layers to enable them to pass their output to the next layer
- First layer specifies the length and width of the input data

### Compile Model 
Compilation of data
- Adam optimizer is used along with the mean squared error loss function 
- Early stopping with a patience of 3 means training will stop if the validation loss does not improve for three epochs in a row

### Train Model
Training of data 
- Trained using the fit() method, passing the training X_train and y_train, number of epocs to train for, batch size, validation data and early stopping callback
- History object contains information about the training process, such as loss and accuracy on the training and validation sets 

## Modifications 
Modifications will be more in-line with the preliminary information researched during Thesis A
- A data preprocessing stage and feature extraction/feature selection will be implemented before the data is used during training and testing
- Potentially a fourth LSTM layer
- Implementation of the activation, normalization and regularization layers (and their respective functions)
- Hyperparameter reserach 
