from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, BatchNormalization
from keras.regularizers import l2

# define the model architecture
model = Sequential()

# add LSTM layer with dropout, activation, normalization and regularisation
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(BatchNormalization())

# add another LSTM layer with dropout, activation, normalization and regularisation
model.add(LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(BatchNormalization())

# add third LSTM layer with dropout, activation, normalization and regularisation
model.add(LSTM(units=64, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(BatchNormalization())

#add fourth LSTM layer with dropout, activation, normalisation and regularisation 
model.add(LSTM(units=64, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(BatchNormalization())

# add output layer
model.add(Dense(units=num_classes, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# set early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=120, validation_data=(X_val, y_val), callbacks=[early_stop])
