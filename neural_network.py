from keras.models import Sequential
from keras.layers import Dense, LSTM

# define the model architecture
model = Sequential()
model.add(LSTM(units=50, input_shape=(timesteps, input_dim)))
model.add(Dense(units=output_dim, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_val, Y_val))
