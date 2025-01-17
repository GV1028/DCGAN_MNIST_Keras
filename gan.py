from keras.models import Sequential
from keras.layers import Input, Merge
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
import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = mnist.load_data()



generator = Sequential()
generator.add(Dense(input_dim=128,output_dim=1024,init='glorot_normal'))
generator.add(BatchNormalization())
generator.add(Activation('tanh'))
generator.add(Dense(128*7*7))
generator.add(BatchNormalization())
generator.add(Activation('tanh'))
generator.add(Reshape((128,7,7)))
generator.add(UpSampling2D(size=(2,2)))
generator.add(Convolution2D(64,5,5,border_mode='same',init='glorot_uniform'))
generator.add(Activation('tanh'))
generator.add(UpSampling2D(size=(2,2)))
generator.add(Convolution2D(32,3,3,border_mode='same',init='glorot_uniform'))
generator.add(Activation('tanh'))
generator.add(Convolution2D(1,1,1,border_mode='same',init='glorot_uniform'))
generator.add(Activation('tanh'))

adamgen = Adam(lr=0.0001)
adamdis = Adam(lr=0.0001)
generator.compile(loss='binary_crossentropy', optimizer = "Adam")




discriminator = Sequential()
discriminator.add(Convolution2D(32,3,3,border_mode='same',input_shape=(1,28,28)))
discriminator.add(LeakyReLU(0.3))
discriminator.add(Convolution2D(32,3,3,border_mode='same'))
discriminator.add(LeakyReLU(0.3))
discriminator.add(MaxPooling2D(pool_size=(2,2)))
discriminator.add(Dropout(0.5))
discriminator.add(Flatten())
discriminator.add(Dense(128))
discriminator.add(LeakyReLU(0.3))
discriminator.add(Dropout(0.5))
discriminator.add(Dense(2))
discriminator.add(Activation('sigmoid'))
discriminator.load_weights("/home/vignesh/Desktop/GAN/discriminator_weights.h5")

discriminator.compile(loss='binary_crossentropy', optimizer = "Adam")



GAN = Sequential()
GAN.add(generator)
GAN.add(discriminator)
GAN.compile(loss='binary_crossentropy', optimizer = "Adam")



def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


nb_epochs = 0
batch_size = 128
dis_loss = []
gan_loss = []

for e in range(nb_epochs):
	random_index = random.sample(range(0,X_train.shape[0]),batch_size)
	mnist_train_data = X_train[random_index,:,:]
	mnist_train_data = mnist_train_data.reshape(batch_size,1,28,28)
	noise = np.random.uniform(-1,1,size=[batch_size,128])
	generated_images = generator.predict(noise) 
	discriminator_train_data = np.concatenate((mnist_train_data,generated_images))
	discriminator_train_labels = np.zeros((2*batch_size))
	discriminator_train_labels[0:batch_size] = 1
	discriminator_train_labels[batch_size:2*batch_size+1] = 0
	discriminator_train_labels = discriminator_train_labels.astype('int')
	discriminator_train_labels = np_utils.to_categorical(discriminator_train_labels, 2)
	make_trainable(discriminator,True)
	#Training Discriminator 
	dl = discriminator.train_on_batch(discriminator_train_data,discriminator_train_labels)
	dis_loss.append(dl)
	gan_data = np.random.uniform(-1,1,size=(batch_size,128))
	gan_labels = np.zeros((batch_size,2))
	gan_labels[:,1] = 1
	make_trainable(discriminator,False)
	#Training GAN
	gl = GAN.train_on_batch(gan_data,gan_labels)
	gan_loss.append(gl)
	if e%100 == 0:
		print "Epoch: %i" %e
		print "Discriminator Loss = %f" %dl 
		print "Generator Loss = %f" %gl
		discriminator.save_weights("/home/vignesh/Desktop/GAN/gandisweights.h5")
		generator.save_weights("/home/vignesh/Desktop/GAN/gangenweights.h5")
		GAN.save_weights("/home/vignesh/Desktop/GAN/ganweights.h5")
		plotlosses(gan_loss,dis_loss,e)
		plot_gen(number = e)
		

