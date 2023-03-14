import pandas as pd
from model import Regression
from config import X_test, X_train
import tensorflow as tf
import matplotlib.pyplot as plt

Predictor = Regression()
Predictor(tf.ones((1, X_train.shape[1])))
Predictor.load_weights('last_model.h5')

y_test = Predictor.predict(X_test)

plt.plot(sorted(y_test))
plt.show()

pd.DataFrame(y_test).to_csv('predictions.csv', header=['target'])
