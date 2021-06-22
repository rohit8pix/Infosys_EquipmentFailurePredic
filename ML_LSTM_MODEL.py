from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix,accuracy_score

nb_features = x_train.shape[2]
timestamp = seq_len
model = Sequential()
 
model.add(LSTM(input_shape = (timestamp, nb_features),units = 50,return_sequences = True))
 
model.add(Dropout(0.4))
 
model.add(LSTM(units = 25,return_sequences = False))
 
model.add(Dropout(0.4))
 
model.add(Dense(units = 1, activation = 'sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
 
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32,validation_split=0.2, shuffle=True, verbose=1)
          #callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
scores = model.evaluate(x_train, y_train, verbose=1, batch_size=32)
print('Accurracy: {}'.format(scores[1]))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
y_pred=model.predict_classes(x_test)
print('Accuracy of model on test data: ',accuracy_score(y_test,y_pred))
print('Confusion Matrix: \n',confusion_matrix(y_test,y_pred))
