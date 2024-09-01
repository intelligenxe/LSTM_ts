This reference explains the fundamentals of the code
Reference: https://pangkh98.medium.com/multi-step-multivariate-time-series-forecasting-using-lstm-92c6d22cd9c2#107e

MODEL

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(n_steps_out))
model.add(Activation('linear'))
model.summary()                                                                                       

BASIC MODEL TRAINING

model.compile(loss='mse' , optimizer=opt , metrics=['mse'])
history = model.fit(train_X , train_y , epochs=60 , steps_per_epoch=25 , verbose=1 ,validation_data=(test_X, test_y) ,shuffle=False)


EXPLANATION OF MODEL AND BASIC TRAINING

This code snippet defines, compiles, and trains a Sequential neural network model using Keras, specifically designed for time series forecasting or sequence prediction using Long Short-Term Memory (LSTM) layers. Let's break down each part:

1. Model Definition
python
Copy code
model = Sequential()
Sequential(): Initializes a Sequential model, which is a linear stack of layers. This is the simplest type of model in Keras, where you can easily add layers one after another.

2. Adding Layers to the Model
python
Copy code
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)):
LSTM(50): Adds an LSTM layer with 50 units (neurons). LSTM layers are useful for processing sequences, making them ideal for time series data or any sequence prediction tasks.
activation='relu': Uses the ReLU (Rectified Linear Unit) activation function for the LSTM units.
return_sequences=True: Ensures that the LSTM layer returns the full sequence output instead of just the last output in the sequence. This is important when stacking LSTM layers, as the next LSTM layer will need the full sequence.
input_shape=(n_steps_in, n_features): Specifies the shape of the input data:
n_steps_in: The number of time steps in the input sequence.
n_features: The number of features per time step.
python
Copy code
model.add(LSTM(50, activation='relu'))
Second LSTM Layer: Another LSTM layer with 50 units and ReLU activation is added, but this time without return_sequences=True, meaning it only returns the output of the last time step.
python
Copy code
model.add(Dense(n_steps_out))
Dense(n_steps_out): Adds a fully connected (dense) layer with n_steps_out units. This layer is typically used to produce the final output. The number of units corresponds to the number of steps in the output sequence.
python
Copy code
model.add(Activation('linear'))
Activation('linear'): Adds a linear activation function to the output layer. This is commonly used in regression tasks where the output is a continuous value.

3. Compiling the Model
python
Copy code
model.compile(loss='mse', optimizer=opt, metrics=['mse'])
loss='mse': Specifies the Mean Squared Error (MSE) as the loss function, which is commonly used for regression tasks.
optimizer=opt: Uses the optimizer defined by the variable opt. The optimizer controls the learning process, such as adjusting the learning rate and updating weights.
metrics=['mse']: Specifies that MSE should be tracked as a metric during training.

4. Summarizing the Model
python
Copy code
model.summary()
model.summary(): Prints a summary of the model, including the number of layers, their types, the output shape of each layer, and the number of parameters (weights) in each layer.

5. Training the Model
python
Copy code
history = model.fit(train_X, train_y, epochs=60, steps_per_epoch=25, verbose=1, validation_data=(test_X, test_y), shuffle=False)
train_X, train_y: The training data, where train_X is the input sequence and train_y is the corresponding target output.
epochs=60: The number of complete passes through the training dataset.
steps_per_epoch=25: Number of steps (batches of samples) per epoch. In this case, the training data is divided into 25 steps.
verbose=1: Controls the verbosity of the training output. 1 provides detailed logs for each epoch.
validation_data=(test_X, test_y): The validation data used to evaluate the model's performance at the end of each epoch.
shuffle=False: Disables shuffling of the training data before each epoch. This is important for time series data, where the order of the data points is meaningful.

Summary
The model consists of two LSTM layers, followed by a Dense output layer with a linear activation function.
The model is compiled using Mean Squared Error as the loss function and an optimizer defined by the variable opt.
The model is then trained on the provided training data for 60 epochs, with a validation set used to monitor the model’s performance during training.


ADVANCED MODEL TRAINING 

model_nc.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.6),
             EarlyStopping(monitor='val_loss', patience=200),
             ModelCheckpoint(filepath='best_model_12.h5', monitor='val_loss', save_best_only=False)]
# fit the model
history2=model_nc.fit(train_X , train_y , epochs=500, batch_size=2, validation_data=(test_X, test_y), verbose=2,
               shuffle=True,callbacks=callbacks) 

EXPLANATION OF ADVANCED MODEL TRAINING

This code snippet is part of training a neural network model using TensorFlow/Keras. Let's break it down step by step:

1. Model Compilation
python
Copy code
model_nc.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
model_nc.compile(): This method configures the model for training by specifying the optimizer and loss function.
optimizer='adam': The Adam optimizer is used for training. It's a popular optimization algorithm that combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp.
loss=tf.keras.losses.MeanSquaredError(): The loss function used is Mean Squared Error (MSE). This is a common loss function for regression problems, where the goal is to minimize the squared difference between the predicted and actual values.

2. Callbacks
python
Copy code
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.6),
    EarlyStopping(monitor='val_loss', patience=200),
    ModelCheckpoint(filepath='best_model_12.h5', monitor='val_loss', save_best_only=False)
]
Callbacks are functions that are executed during training at certain points. The ones used here are:

ReduceLROnPlateau:

Purpose: Reduces the learning rate when a metric (in this case, val_loss) has stopped improving.
monitor='val_loss': It monitors the validation loss.
patience=5: If the validation loss doesn't improve for 5 consecutive epochs, the learning rate is reduced.
factor=0.6: The learning rate is reduced by a factor of 0.6.
verbose=1: Provides detailed output when the learning rate is reduced.
EarlyStopping:

Purpose: Stops training when the monitored metric (in this case, val_loss) has stopped improving to prevent overfitting.
monitor='val_loss': It monitors the validation loss.
patience=200: Training will stop if the validation loss doesn't improve for 200 consecutive epochs.
ModelCheckpoint:

Purpose: Saves the model during training.
filepath='best_model_12.h5': Specifies the file path where the model will be saved.
monitor='val_loss': It monitors the validation loss.
save_best_only=False: The model is saved after every epoch, not just when the validation loss improves.

3. Model Training (Fitting)
python
Copy code
history2 = model_nc.fit(train_X, train_y, epochs=500, batch_size=2, validation_data=(test_X, test_y), verbose=2,
                        shuffle=True, callbacks=callbacks)
model_nc.fit(): This method trains the model using the training data.
train_X, train_y: These are the input features and labels for the training set.
epochs=500: The model will go through the entire training dataset 500 times.
batch_size=2: The model will update its weights after processing 2 samples.
validation_data=(test_X, test_y): This specifies the validation data, which the model will use to evaluate performance after each epoch.
verbose=2: This controls the verbosity of the output during training. verbose=2 means one line per epoch is printed.
shuffle=True: The training data is shuffled before each epoch.
callbacks=callbacks: The list of callbacks defined earlier is passed to the training process, allowing the learning rate reduction, early stopping, and model checkpointing to occur as specified.

Summary
The model is compiled with the Adam optimizer and Mean Squared Error loss.
Several callbacks are set up to handle learning rate reduction, early stopping, and model checkpointing based on the validation loss.
The model is trained over 500 epochs with a small batch size of 2, using the training data, and validated on the test data after each epoch. The training process is monitored and adjusted dynamically using the defined callbacks.






