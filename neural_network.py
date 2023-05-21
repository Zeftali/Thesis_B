import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

def prepare_data():
    # load the data from an Excel file 
    data = pd.read_csv('Mirai.csv')

    # drop null values
    data.dropna(inplace=True)

    # encode categorical variables
    data = pd.get_dummies(data)

    # select numeric columns only
    data = data.select_dtypes(include=np.number)

    # split data into X and y
    target_cols = ['Tot_Fwd_Pkts', 'Tot_Bwd_Pkts', 'Flow_Pkts/s', 'Flow_Byts/s', 'Flow_Duration']
    X = data.drop(columns=['Idle_Mean', 'Idle_Max', 'Idle_Min', 'Init_Bwd_Win_Byts', 'Subflow_Bwd_Pkts'], axis=1).values
    y = data[target_cols].values

    # PCA for feature extraction 
    pca = PCA(n_components=1)
    X = pca.fit_transform(X)

    # split the data into training and validation sets 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # get number of input features and output classes 
    num_input_features = X_train.shape[1]
    num_classes = len(target_cols)

    return X_train.reshape(-1, 1, num_input_features), y_train, X_val.reshape(-1, 1, num_input_features), y_val, num_input_features, num_classes

# define the model architecture
def define_model(num_input_features, num_classes):
    model = Sequential()

    # add LSTM layer with dropout, activation, normalization and regularisation
    model.add(LSTM(units=128, input_shape = (1, num_input_features), return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # add another LSTM layer with dropout, activation, normalization and regularisation
    model.add(LSTM(units=128, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # add third LSTM layer with dropout, activation, normalization and regularisation
    model.add(LSTM(units=128, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # add fourth LSTM layer with dropout, activation, normalization and regularisation
    model.add(LSTM(units=128, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # add output layer
    model.add(Dense(units=num_classes, activation='softmax'))

    # compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_logarithmic_error', metrics=['accuracy'])

    return model

def train_model():
    X_train, y_train, X_val, y_val, num_input_features, num_classes = prepare_data()

    # define the model
    model = define_model(num_input_features, num_classes)

    # set early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # train the model
    model.fit(X_train, y_train, batch_size=256, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stop])
    return model


if __name__ == '__main__':
    # train the model and get the trained model
    model = train_model()
