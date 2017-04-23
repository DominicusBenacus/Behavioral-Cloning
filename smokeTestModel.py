model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

#trian the model
from sklearn.utils import shuffle
#X_train, y_train = shuffle(X_train, y_train, random_state=348202)

X_train_small = X_train[0:1000]
y_train_small = y_train[0:1000]
#model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)
model.fit(X_train_small,y_train_small,validation_split=0.2,shuffle=True,nb_epoch=3)

model.save('model.h5')