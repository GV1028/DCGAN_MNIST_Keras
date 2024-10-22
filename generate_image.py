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

#Creating the generator model that inputs a noise from a known distribution and outputs 28x28 image



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
generator.compile(loss='binary_crossentropy', optimizer = "Adam")
generator.load_weights("/home/vignesh/Desktop/GAN/gangenweights.h5")

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
discriminator.compile(loss='binary_crossentropy', optimizer = "Adam")
discriminator.load_weights("/home/vignesh/Desktop/GAN/gandisweights.h5")




#Generating Images for training the discriminator
BATCH_SIZE = 100
noise = np.random.uniform(-1,1,size=[BATCH_SIZE,128])


#Generating images without training the generator so as to create the dataset for discriminator
image = generator.predict(noise)


for i in range(BATCH_SIZE):
	image[i,:,:,:] = image[i,:,:,:]*127.5 + 127.5
	Image.fromarray(image[i,0,:,:].astype(np.uint8)).save("/home/vignesh/Desktop/GAN/ganimages/"+str(i)+".png") 

