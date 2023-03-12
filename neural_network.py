from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

# define the model architecture
model = Sequential()

# add LSTM layer with dropout
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

# add another LSTM layer with dropout
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))

# add third LSTM layer with dropout
model.add(LSTM(units=64))
model.add(Dropout(0.2))

# add output layer
model.add(Dense(units=1))

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# set early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])
