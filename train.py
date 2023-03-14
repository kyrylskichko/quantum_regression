from model import Regression, RMSE
from config import *
from tensorflow import keras

my_model = Regression()

optimizer = keras.optimizers.Adam(learning_rate=lr)
my_model.compile(optimizer=optimizer, loss=RMSE)

history = my_model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_split=val_size,
)

my_model.save_weights('last_model.h5')

