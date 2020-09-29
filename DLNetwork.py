
# For 3D convolutional Networks -------------------------------------------------------
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D

from keras.models import Model
from keras.layers import Dropout, Input
from keras.layers import Flatten, add, Concatenate, GlobalAveragePooling3D, Lambda
from keras.layers import Dense

from keras_pyramid_pooling_module import PyramidPoolingModule
from keras.layers import BatchNormalization
from keras.layers import Activation 
from keras import regularizers
from keras import backend
import keras.backend as K
import numpy as np


IMAGE_ORDERING_CHANNELS_LAST = "channels_last"
IMAGE_ORDERING_CHANNELS_FIRST = "channels_first"

# Default IMAGE_ORDERING = channels_first
IMAGE_ORDERING = IMAGE_ORDERING_CHANNELS_FIRST

if backend.image_data_format() == 'channels_last':
	bn_axis = 3
else:
	bn_axis = 1


def identity_Block(inpt, nb_filter, kernel_size, strides=1, with_conv_shortcut=False):
	# ---------   
	x = Conv3D(nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same', data_format='channels_first') (inpt) 
	# , kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01) 
	x = BatchNormalization(axis=bn_axis)(x)
	x = Activation('relu')(x)
	# ---------
	x = Conv3D(nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same', data_format='channels_first') (x)
	x = BatchNormalization(axis=bn_axis)(x)
	
	if with_conv_shortcut:
		shortcut = Conv3D(nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same', 
				  data_format='channels_first') (inpt)
		shortcut = BatchNormalization(axis=bn_axis)(shortcut)
		
		x = Dropout(0.2)(x)
		x = add([x, shortcut])
		x = Activation('relu')(x)
        
		#x = Conv3D(nb_filter=nb_filter, kernel_size=(1, 1, 1), strides=1, padding='same',data_format='channels_first')(x)  # Bottle neck
		#x = BatchNormalization(axis=bn_axis)(x)  
		#x = Activation('relu')(x)          
		return x
	else:
		x = add([x, inpt])
		x = Activation('relu')(x)
        
		#x = Conv3D(nb_filter=nb_filter, kernel_size=(1, 1, 1), strides=1, padding='same',data_format='channels_first')(x)  # Bottle neck
		#x = BatchNormalization(axis=bn_axis)(x)  
		#x = Activation('relu')(x)        
		return x

# Network-----
def net_structure(inpt):
	k1 = 12#16#12
	k2 = 24#32#24
	k3 = 36#64#36
	k4 = 48#128#48       

	x1 = ZeroPadding3D((1, 1, 1), data_format='channels_first')(inpt) 
	x1 = Conv3D(nb_filter=k1, kernel_size=(3, 3, 3), strides=1, padding='valid',data_format='channels_first')(x1) 
	x1 = Conv3D(nb_filter=k1, kernel_size=(3, 3, 3), strides=1, padding='same',data_format='channels_first')(x1)     
	x1 = BatchNormalization()(x1)
	x1 = Activation('relu')(x1)


	x1 = identity_Block(x1, nb_filter=k2, kernel_size=(3, 3, 3), strides=1, with_conv_shortcut=True)
	x1 = identity_Block(x1, nb_filter=k2, kernel_size=(3, 3, 3))
	x1 = AveragePooling3D(pool_size=(2, 2, 1), strides=(2,2,1), data_format='channels_first')(x1)

	x1 = identity_Block(x1, nb_filter=k3, kernel_size=(3, 3, 3), strides=1, with_conv_shortcut=True)
	x1 = identity_Block(x1, nb_filter=k3, kernel_size=(3, 3, 3))  
	x1 = AveragePooling3D(pool_size=(2, 2, 1), strides=(2,2,1), data_format='channels_first')(x1)

	x1 = identity_Block(x1, nb_filter=k4, kernel_size=(3, 3, 3), strides=1, with_conv_shortcut=True)
	x1 = identity_Block(x1, nb_filter=k4, kernel_size=(3, 3, 3))
	x1 = AveragePooling3D(pool_size=(2, 2, 2), strides=(2,2,2), data_format='channels_first')(x1)   # ***(2,2,1)
	x1 = Flatten()(x1)
    
	return x1

# main ---
def Generate_DLNetwork(shape, classes):
	
	inpt_1 = Input(shape=shape)
	x1 = net_structure(inpt_1)
    
	inpt_2 = Input(shape=shape)
	x2 = net_structure(inpt_2)
    
	inpt_3 = Input(shape=shape)
	x3 = net_structure(inpt_3)
    
	inpt_4 = Input(shape=shape)
	x4 = net_structure(inpt_4)
    
	inpt_5 = Input(shape=shape)
	x5 = net_structure(inpt_5)
    
	inpt_6 = Input(shape=shape)
	x6 = net_structure(inpt_6)    

	# --------------------------------Merge All Inputs:--------------------------------------       
	output = Concatenate(axis=1)([x1, x2, x3, x4, x5, x6]) 
	output = Dense(100, activation='relu')(output)    
	output = Dropout(0.25)(output)       
	output = Dense(classes, activation='softmax')(output) # 
	  
	model = Model(inputs=[inpt_1, inpt_2, inpt_3, inpt_4, inpt_5, inpt_6], outputs=output)

    
   
	return model







