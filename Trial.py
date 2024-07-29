from MyTorch import MLP
from keras.datasets import boston_housing

import warnings
warnings.filterwarnings("ignore")

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# normalize the data
x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

nn = MLP(13, [64, 32, 1])
nn.training(x_train, y_train, x_test, y_test)