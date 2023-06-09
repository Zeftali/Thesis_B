import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler

def prepare_data():
    # load the data from an Excel file 
    data = pd.read_csv('IoT-DS2.csv')
    
    # drop null values
    data.dropna(inplace=True)
    
    # encode categorical variables
    data = pd.get_dummies(data)

    # split data into X and y
    target_col = 'Cat'
    X = data.drop(columns=[target_col], axis=1).values
    y = data[target_col].map({'Normal': 0, 'Mirai': 1}).values

    # num of classes for binary classification
    num_classes = len(np.unique(y)) 

    # convert y to numpy array and reshape for compatibility with the model
    y = np.array(y).reshape(-1, 1)

    # normalize the input features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # PCA for feature extraction. Determine the number of components using explained variance ratio.
    pca = PCA(n_components=4)
    X = pca.fit_transform(X)

    # Shuffle the data
    X, y = shuffle(X, y, random_state=None)

    # Apply Random Over-Sampling
    oversampler = RandomOverSampler()
    X, y = oversampler.fit_resample(X, y)

    # split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=None)

    # get the number of input features and output classes
    num_input_features = X_train.shape[2]

    return (
        X_train.reshape(-1, 1, num_input_features),
        y_train,
        X_val.reshape(-1, 1, num_input_features),
        y_val,
        num_input_features,
        num_classes,
    )

# define the model architecture
def define_model(num_input_features, num_classes):
    model = Sequential()
    # add LSTM layer with dropout, activation, normalization and regularisation
    model.add(LSTM(units=128, input_shape=(1, num_input_features), return_sequences=True, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # add another LSTM layer with dropout, activation, normalization and regularisation
    model.add(LSTM(units=128, return_sequences=True, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # add third LSTM layer with dropout, activation, normalization and regularisation
    model.add(LSTM(units=128, return_sequences=True, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # add fourth LSTM layer with dropout, activation, normalization and regularisation
    model.add(LSTM(units=128, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # add output layer
    model.add(Dense(units=num_classes, activation='sigmoid'))

    # compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_model():
    X_train, y_train, X_val, y_val, num_input_features, num_classes = prepare_data()
    # define the model
    model = define_model(num_input_features, num_classes)
    # set early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    # train the model
    history = model.fit(X_train, y_train, batch_size=128, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stop])
   
   # plot the training and validation metrics
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    # evaluate the model on the validation set
    y_val_pred_prob = model.predict(X_val)
    y_val_pred = (y_val_pred_prob > 0.5).astype(int)

    # convert y_val to numpy array and reshape for compatibility with classification metrics
    y_val = np.array(y_val).reshape(-1)

    # compute classification metrics
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1))
    
    # generate classification report and confusion matrix
    report = classification_report(y_val, y_val_pred)
    matrix = confusion_matrix(y_val, y_val_pred)

    print("Classification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(matrix)

    return model

if __name__ == '__main__':
    # train the model and get the trained model
    model = train_model()
