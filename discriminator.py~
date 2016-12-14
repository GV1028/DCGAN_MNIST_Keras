from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization 
import numpy as np
from keras.utils.visualize_util import plot
from keras.models import Model
from PIL import Image
from keras.datasets import mnist
from scipy import misc
from keras.utils import np_utils
import random

#Importing the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


#Creating data to pre-train the discriminator
mnist_train = 10000
random_index = random.sample(range(0,X_train.shape[0]),mnist_train)
mnist_train_data = X_train[random_index,:,:]
generated_images = np.zeros((10000,28,28))
for i in range(10000):
	generated_images[i,:,:] = misc.imread("/home/vignesh/Desktop/GAN/generatedimages/" + str(i) + ".png")
X_train = np.concatenate((mnist_train_data,generated_images))
Y_train = np.zeros((20000))
Y_train[0:10000] = 1
Y_train[10000:20000] = 0
X_train = X_train.reshape((20000,1,28,28))
Y_train = Y_train.astype('int')
Y_train = np_utils.to_categorical(Y_train, 2)

#Creating the discriminator model
discriminator = Sequential()
discriminator.add(Convolution2D(32,3,3,border_mode='same',input_shape=(1,28,28)))
discriminator.add(LeakyReLU(0.3))
discriminator.add(Convolution2D(32,3,3,border_mode='same'))
discriminator.add(LeakyReLU(0.3))
discriminator.add(MaxPooling2D(pool_size=(2,2)))
discriminator.add(Dropout(0.25))
discriminator.add(Flatten())
discriminator.add(Dense(128))
discriminator.add(LeakyReLU(0.3))
discriminator.add(Dropout(0.5))
discriminator.add(Dense(2))
discriminator.add(Activation('sigmoid'))
discriminator.summary()

#Training
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)	 
discriminator.compile(loss='binary_crossentropy', optimizer = "adam",metrics=['accuracy'])
discriminator.fit(X_train,Y_train,batch_size=128,nb_epoch=1)
discriminator.save_weights("/home/vignesh/Desktop/GAN/discriminator_weights.h5",overwrite=False)
