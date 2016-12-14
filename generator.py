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
generator.add(LeakyReLU(alpha=0.1))
generator.add(Dense(128*7*7))
generator.add(BatchNormalization())
generator.add(LeakyReLU(alpha=0.1))
generator.add(Reshape((128,7,7)))
generator.add(UpSampling2D(size=(2,2)))
generator.add(Convolution2D(64,5,5,border_mode='same',init='glorot_uniform'))
generator.add(LeakyReLU(alpha=0.1))
generator.add(UpSampling2D(size=(2,2)))
generator.add(Convolution2D(32,3,3,border_mode='same',init='glorot_uniform'))
generator.add(LeakyReLU(alpha=0.1))
generator.add(Convolution2D(1,1,1,border_mode='same',init='glorot_uniform'))
generator.add(Activation('tanh'))
generator.summary()

#Generating Images for training the discriminator
BATCH_SIZE = 10
noise = np.random.uniform(-1,1,size=[BATCH_SIZE,128])

#Define optimizer and compile the generator model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)	 
generator.compile(loss='binary_crossentropy', optimizer = "adam")
generator.load_weights("/home/vignesh/Desktop/GAN/gangenweights.h5")

#Generating images without training the generator so as to create the dataset for discriminator
image = generator.predict(noise)
for i in range(BATCH_SIZE):
	image[i,:,:,:] = image[i,:,:,:]*127.5 + 127.5
	Image.fromarray(image[i,0,:,:].astype(np.uint8)).save("/home/vignesh/Desktop/GAN/ganimages/"+str(i)+".png") 




