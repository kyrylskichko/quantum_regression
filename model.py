from tensorflow import keras
from keras import backend as K
from config import X_train


def RMSE(y_t, y_p):
    return K.sqrt(K.mean(K.square(y_p - y_t)))

class Regression(keras.Model):
    def __init__(self):
        super(Regression, self).__init__()
        self.norm = keras.layers.Normalization()
        self.dense = keras.layers.Dense(1, activation='linear', input_shape=(X_train.shape[1],))

    def call(self, x):
        x = self.norm(x)
        x = self.dense(x)

        return x



