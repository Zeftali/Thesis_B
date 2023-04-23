import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA

def prepare_data():
    # load the data from the CSV file 
    data = pd.read_csv('Mirai.csv')
    
    # drop null values
    data.dropna(inplace=True)

    # encode categorical variables
    data = pd.get_dummies(data)

    # select numeric columns only
    data = data.select_dtypes(include=np.number)

    # split data into X and y
    target_col = 'Flow_Duration'
    X = data.drop(columns=[target_col], axis=1).values
    y = data[target_col].values
    
    # reshape X into a 3D array with shape (n_samples, 2, num_input_features)
    X = X.reshape(-1, 2, X.shape[1])

    # split the data into training and validation sets 
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    # get number of input features and output classes 
    num_input_features = X_train.shape[2]
    num_classes = len(set(y))

    return X_train, y_train, X_val, y_val, num_input_features, num_classes


# define the model architecture
def define_model(num_input_features, num_classes):
    model = Sequential()

    # add LSTM layer with dropout, activation, normalization and regularisation
    model.add(LSTM(units=64, input_shape=(2, num_input_features), return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # add another LSTM layer with dropout, activation, normalization and regularisation
    model.add(LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # add third LSTM layer with dropout, activation, normalization and regularisation
    model.add(LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    #add fourth LSTM layer with dropout, activation, normalisation and regularisation 
    model.add(LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # add output layer
    model.add(Dense(units=num_classes, activation='softmax'))

    # compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model():
    X_train, y_train, X_val, y_val, num_input_features, num_classes = prepare_data()

    # define the model
    model = define_model(num_input_features, num_classes)

    # set early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # train the model
    model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stop])
    return model


if __name__ == '__main__':
    # train the model and get the trained model
    model = train_model()

    # evaluate the model on validation data and print accuracy
    X_train, y_train, X_val, y_val, num_input_features, num_classes = prepare_data()
    score = model.evaluate(X_val.reshape(-1, 1, num_input_features), y_val, verbose=0)
    print('Validation accuracy:', score[1])