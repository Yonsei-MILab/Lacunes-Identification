"""
Created on Monday June 22 2020

@author: Mohammed Al-masni
"""
###############################
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils, to_categorical,plot_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from nilearn import image

from sklearn import preprocessing
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras import backend as K
from keras.backend import categorical_crossentropy

from read_data import read_data
from DLNetwork import Generate_DLNetwork
import math
# -------------------------------------------------------

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

#------------------------------------------------------------------------------
Train = 1    
Test  = 1 

epoch = 100
learningRate = 0.0001 
decay = learningRate/epoch
optimizer = Adam(lr=learningRate)
batch_size = 50

Height = 32
Width  = 32
Depth  = 5
shape  = [1, Height, Width, Depth]
n_classes   = 2 # binary classification: 0: tissue, 1: lesion


#-- This part to save and load the MODEL weights:------------------------------
def save_model(model,md = 'lstm'):
	model_json = model.to_json()
	with open("model_"+md+".json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("model_"+md+".h5")
	print("The model is successfully saved")

def load_model(md = 'lstm'):
	json_file = open("model_"+md+".json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model_"+md+".h5")
	print("Loaded model from disk")
	return loaded_model

def scheduler(epoch):
    ep = 10
    #return learningRate * math.exp(0.1 * (-epoch))
    if epoch < ep:
        return learningRate
    else:
        return learningRate * math.exp(0.1 * (ep - epoch)) 

# ======================================================================================================
def main():
	class_names = ['Non-Lacunes', 'Lacunes']
	class_labels = [0,1]
	
	x_train_flair_32x, x_train_flair_48x, x_train_flair_64x, x_train_T1_32x, x_train_T1_48x, x_train_T1_64x, y_test_flair_32x, y_test_flair_48x, y_test_flair_64x, y_test_T1_32x, y_test_T1_48x, y_test_T1_64x, y_valid_flair_32x, y_valid_flair_48x, y_valid_flair_64x, y_valid_T1_32x, y_valid_T1_48x, y_valid_T1_64x, tr_onehot_flair_32x, tr_onehot_flair_48x, tr_onehot_flair_64x, tr_onehot_T1_32x, tr_onehot_T1_48x, tr_onehot_T1_64x, ts_onehot_flair_32x, ts_onehot_flair_48x, ts_onehot_flair_64x, ts_onehot_T1_32x, ts_onehot_T1_48x, ts_onehot_T1_64x, val_onehot_flair_32x, val_onehot_flair_48x, val_onehot_flair_64x, val_onehot_T1_32x, val_onehot_T1_48x, val_onehot_T1_64x = read_data(class_names,class_labels)
	
		    
	print('---------------------------------')
	print('Trainingdata=',x_train_flair_32x.shape)
	print('Validationdata=',y_valid_flair_32x.shape)
	print('Testingdata=',y_test_flair_32x.shape)
	print('Traininglabel=',tr_onehot_flair_32x.shape)
	print('Validationlabel=',val_onehot_flair_32x.shape)
	print('Testinglabel=',ts_onehot_flair_32x.shape)
	print('---------------------------------')   
	
	if Train:
		print('Generating DL Model...')
		model = Generate_DLNetwork(shape, n_classes)
		model.summary()
		
		model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy']) 
		print('Training Model')
		csv_logger = CSVLogger('Loss_Acc.csv', append=True, separator=' ')
		checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
		#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')
		reduce_lr = LearningRateScheduler(scheduler)
		           
		model.fit(x = [x_train_flair_32x, x_train_flair_48x, x_train_flair_64x, x_train_T1_32x, x_train_T1_48x, x_train_T1_64x],           
			y = tr_onehot_flair_32x, # can be one GT label for six inputs
			batch_size = batch_size,
			shuffle=True,
			epochs = epoch, 
			verbose = 1,          # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
			validation_data=([y_valid_flair_48x, y_valid_flair_64x, y_valid_T1_48x, y_valid_T1_64x],             
			#validation_data=([y_valid_flair_32x, y_valid_flair_64x, y_valid_T1_32x, y_valid_T1_64x],             
			#validation_data=([y_valid_flair_32x, y_valid_flair_48x, y_valid_T1_32x, y_valid_T1_48x],            
			#validation_data=([y_valid_flair_64x, y_valid_T1_64x],                  
			#validation_data=([y_valid_flair_48x, y_valid_T1_48x],                     
			#validation_data=([y_valid_flair_32x, y_valid_T1_32x],                  
			#validation_data=([y_valid_flair_32x, y_valid_flair_48x, y_valid_flair_64x, y_valid_T1_32x, y_valid_T1_48x, y_valid_T1_64x],     
			val_onehot_flair_32x),
			callbacks=[csv_logger, checkpoint, reduce_lr]) 
	
		save_model(model,'Lacune_ResNet_') # to save the WEIGht 
	
	if Test:
		# Load the model and make file with predicted labels ##
		new_model = load_model('Lacune_ResNet_') # to load the weight
		new_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy']) 
   
		pred = new_model.predict([y_test_flair_32x,  y_test_flair_48x, y_test_flair_64x, y_test_T1_32x, y_test_T1_48x, y_test_T1_64x])
		
		# Evaluate the result and show an example ##
		print("Evaluate Model ...")
		
		# To measure the all evaluation metrics:-------------------------------  
		print('Data predicted for all model')
		Actual_label_test    = np.argmax(ts_onehot_flair_32x,axis=1) # to make all class labels in one colum # argmax provide position of maximum value   ???
		Predicted_label_test = np.argmax(np.round(pred),axis=1)
		np.savetxt("path/filename.csv", Actual_label_test, fmt='%.6f', delimiter=',') 
		np.savetxt("path/filename.csv", Predicted_label_test, fmt='%.6f', delimiter=',')
		
if __name__ == "__main__":
	main()  
	
	
	
