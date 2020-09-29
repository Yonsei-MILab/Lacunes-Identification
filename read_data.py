import glob
from PIL import Image
import numpy as np
from keras.utils import np_utils, to_categorical
from nilearn import image

Height = 32
Width  = 32
Depth  = 5
shape  = [1, Height, Width, Depth]

def read_data(class_names,class_labels):
	fold1_test_flair_32x  = list()  # Testing data for all classes
	fold1_test_flair_48x  = list()  
	fold1_test_flair_64x  = list()  
	fold1_test_T1_32x = list()  # Testing data for all classes   
	fold1_test_T1_48x = list()   
	fold1_test_T1_64x = list() 
	ts_lbl_flair_32x = list()  # Testing Labels for all classes
	ts_lbl_flair_48x = list()  
	ts_lbl_flair_64x = list()  
	ts_lbl_T1_32x = list()  # Testing Labels for all classes
	ts_lbl_T1_48x = list()  
	ts_lbl_T1_64x = list()  
	
	fold2_train_flair_32x  = list()  # Training data for all classes
	fold2_train_flair_48x  = list()  
	fold2_train_flair_64x  = list()  
	fold2_train_T1_32x = list()  # Training data for all classes   
	fold2_train_T1_48x = list()   
	fold2_train_T1_64x = list() 
	tr_lbl_flair_32x = list()  # Training Labels for all classes
	tr_lbl_flair_48x = list()  
	tr_lbl_flair_64x = list()  
	tr_lbl_T1_32x = list()  # Training Labels for all classes
	tr_lbl_T1_48x = list()  
	tr_lbl_T1_64x = list()      
	 
	fold3_valid_flair_32x  = list()  # Validation data for all classes
	fold3_valid_flair_48x  = list()  
	fold3_valid_flair_64x  = list()  
	fold3_valid_T1_32x = list()  # Validation data for all classes   
	fold3_valid_T1_48x = list()   
	fold3_valid_T1_64x = list()
	val_lbl_flair_32x = list()  # Validation Labels for all classes
	val_lbl_flair_48x = list()  
	val_lbl_flair_64x = list()  
	val_lbl_T1_32x = list()  # Validation Labels for all classes
	val_lbl_T1_48x = list()  
	val_lbl_T1_64x = list()        
	
	path = '/media/milab/4TB/4TB_Backup/Almasni/Lacune/Fold 1/'

	for pos,sel in enumerate(class_names):
		#print(sel)
		# 1st Input
		images_train_Flair_32x  = sorted(glob.glob(path+"Training/"+sel+"/01 32x32x5-Flair/*.nii"))
		images_valid_Flair_32x  = sorted(glob.glob(path+"Validation/"+sel+"/01 32x32x5-Flair/*.nii")) 
		images_test_Flair_32x   = sorted(glob.glob(path+"Testing/"+sel+"/01 32x32x5-Flair/*.nii"))       
		# 2nd Input
		images_train_Flair_48x  = sorted(glob.glob(path+"Training/"+sel+"/02 resized from 48x48x5-Flair/*.nii"))
		images_valid_Flair_48x  = sorted(glob.glob(path+"Validation/"+sel+"/02 resized from 48x48x5-Flair/*.nii")) 
		images_test_Flair_48x   = sorted(glob.glob(path+"Testing/"+sel+"/02 resized from 48x48x5-Flair/*.nii"))        
		# 3rd Input
		images_train_Flair_64x  = sorted(glob.glob(path+"Training/"+sel+"/03 resized from 64x64x5-Flair/*.nii"))
		images_valid_Flair_64x  = sorted(glob.glob(path+"Validation/"+sel+"/03 resized from 64x64x5-Flair/*.nii")) 
		images_test_Flair_64x   = sorted(glob.glob(path+"Testing/"+sel+"/03 resized from 64x64x5-Flair/*.nii"))      
		# 4th Input
		images_train_T1_32x  = sorted(glob.glob(path+"Training/"+sel+"/01 32x32x5-T1/*.nii"))
		images_valid_T1_32x  = sorted(glob.glob(path+"Validation/"+sel+"/01 32x32x5-T1/*.nii")) 
		images_test_T1_32x   = sorted(glob.glob(path+"Testing/"+sel+"/01 32x32x5-T1/*.nii"))       
		# 5th Input
		images_train_T1_48x  = sorted(glob.glob(path+"Training/"+sel+"/02 resized from 48x48x5-T1/*.nii"))
		images_valid_T1_48x  = sorted(glob.glob(path+"Validation/"+sel+"/02 resized from 48x48x5-T1/*.nii")) 
		images_test_T1_48x   = sorted(glob.glob(path+"Testing/"+sel+"/02 resized from 48x48x5-T1/*.nii"))        
		# 6th Input
		images_train_T1_64x  = sorted(glob.glob(path+"Training/"+sel+"/03 resized from 64x64x5-T1/*.nii"))
		images_valid_T1_64x  = sorted(glob.glob(path+"Validation/"+sel+"/03 resized from 64x64x5-T1/*.nii")) 
		images_test_T1_64x   = sorted(glob.glob(path+"Testing/"+sel+"/03 resized from 64x64x5-T1/*.nii"))              

		#Testing database:-------------------------------------------------------------
		# 1st Input
		for volume in images_test_Flair_32x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold1_test_flair_32x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			ts_lbl_flair_32x.append(class_labels[pos])
		# 2nd Input
		for volume in images_test_Flair_48x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold1_test_flair_48x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			ts_lbl_flair_48x.append(class_labels[pos])            
		# 3rd Input
		for volume in images_test_Flair_64x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold1_test_flair_64x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			ts_lbl_flair_64x.append(class_labels[pos])         
		# 4th Input
		for volume in images_test_T1_32x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold1_test_T1_32x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			ts_lbl_T1_32x.append(class_labels[pos])      
		# 5th Input
		for volume in images_test_T1_48x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold1_test_T1_48x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			ts_lbl_T1_48x.append(class_labels[pos])                   
		# 6th Input
		for volume in images_test_T1_64x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold1_test_T1_64x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			ts_lbl_T1_64x.append(class_labels[pos])              
		   
		#Training database:-------------------------------------------------------------
		# 1st Input
		for volume in images_train_Flair_32x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold2_train_flair_32x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			tr_lbl_flair_32x.append(class_labels[pos])
		# 2nd Input
		for volume in images_train_Flair_48x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold2_train_flair_48x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			tr_lbl_flair_48x.append(class_labels[pos])            
		# 3rd Input
		for volume in images_train_Flair_64x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold2_train_flair_64x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			tr_lbl_flair_64x.append(class_labels[pos])         
		# 4th Input
		for volume in images_train_T1_32x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold2_train_T1_32x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			tr_lbl_T1_32x.append(class_labels[pos])      
		# 5th Input
		for volume in images_train_T1_48x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold2_train_T1_48x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			tr_lbl_T1_48x.append(class_labels[pos])                   
		# 6th Input
		for volume in images_train_T1_64x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold2_train_T1_64x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			tr_lbl_T1_64x.append(class_labels[pos])  

		#Validation database:-------------------------------------------------------------
		# 1st Input
		for volume in images_valid_Flair_32x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold3_valid_flair_32x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			val_lbl_flair_32x.append(class_labels[pos])
		# 2nd Input
		for volume in images_valid_Flair_48x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold3_valid_flair_48x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			val_lbl_flair_48x.append(class_labels[pos])            
		# 3rd Input
		for volume in images_valid_Flair_64x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold3_valid_flair_64x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			val_lbl_flair_64x.append(class_labels[pos])         
		# 4th Input
		for volume in images_valid_T1_32x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold3_valid_T1_32x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			val_lbl_T1_32x.append(class_labels[pos])      
		# 5th Input
		for volume in images_valid_T1_48x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold3_valid_T1_48x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			val_lbl_T1_48x.append(class_labels[pos])                   
		# 6th Input
		for volume in images_valid_T1_64x:
			img = image.load_img(volume)   # *****---> may it need to convert to arrray!!!!
			img = img.get_data()    # convert to array
			fold3_valid_T1_64x.append(np.asarray(img, dtype = np.float32) / 1.)   #65535.
			val_lbl_T1_64x.append(class_labels[pos])              
   
	print('DataShape=',img.shape)      
	x_train_flair_32x = np.asarray(fold2_train_flair_32x)
	x_train_flair_48x = np.asarray(fold2_train_flair_48x)
	x_train_flair_64x = np.asarray(fold2_train_flair_64x)
	x_train_T1_32x = np.asarray(fold2_train_T1_32x)
	x_train_T1_48x = np.asarray(fold2_train_T1_48x)
	x_train_T1_64x = np.asarray(fold2_train_T1_64x)   

	y_test_flair_32x = np.asarray(fold1_test_flair_32x)
	y_test_flair_48x = np.asarray(fold1_test_flair_48x)
	x_test_flair_64x = np.asarray(fold1_test_flair_64x)    
	y_test_T1_32x = np.asarray(fold1_test_T1_32x)
	y_test_T1_48x = np.asarray(fold1_test_T1_48x)
	y_test_T1_64x = np.asarray(fold1_test_T1_64x) 
	
	y_valid_flair_32x = np.asarray(fold3_valid_flair_32x)
	y_valid_flair_48x = np.asarray(fold3_valid_flair_48x)
	x_valid_flair_64x = np.asarray(fold3_valid_flair_64x)    
	y_valid_T1_32x = np.asarray(fold3_valid_T1_32x)
	y_valid_T1_48x = np.asarray(fold3_valid_T1_48x)
	y_valid_T1_64x = np.asarray(fold3_valid_T1_64x)     
	
	tr_lbl_flair_32x  = np.asarray(tr_lbl_flair_32x)    
	tr_lbl_flair_48x  = np.asarray(tr_lbl_flair_48x)    
	tr_lbl_flair_64x  = np.asarray(tr_lbl_flair_64x) 
	tr_lbl_T1_32x  = np.asarray(tr_lbl_T1_32x)    
	tr_lbl_T1_48x  = np.asarray(tr_lbl_T1_48x)    
	tr_lbl_T1_64x  = np.asarray(tr_lbl_T1_64x)     

	ts_lbl_flair_32x  = np.asarray(ts_lbl_flair_32x)    
	ts_lbl_flair_48x  = np.asarray(ts_lbl_flair_48x)    
	ts_lbl_flair_64x  = np.asarray(ts_lbl_flair_64x) 
	ts_lbl_T1_32x  = np.asarray(ts_lbl_T1_32x)    
	ts_lbl_T1_48x  = np.asarray(ts_lbl_T1_48x)    
	ts_lbl_T1_64x  = np.asarray(ts_lbl_T1_64x)    
	
	val_lbl_flair_32x  = np.asarray(val_lbl_flair_32x)    
	val_lbl_flair_48x  = np.asarray(val_lbl_flair_48x)    
	val_lbl_flair_64x  = np.asarray(val_lbl_flair_64x) 
	val_lbl_T1_32x  = np.asarray(val_lbl_T1_32x)    
	val_lbl_T1_48x  = np.asarray(val_lbl_T1_48x)    
	val_lbl_T1_64x  = np.asarray(val_lbl_T1_64x)  
	
	# in case of 'channels_first'    
	x_train_flair_32x = x_train_flair_32x.reshape(x_train_flair_32x.shape[0], 1, Height, Width, Depth)
	x_train_flair_48x = x_train_flair_48x.reshape(x_train_flair_48x.shape[0], 1, Height, Width, Depth)    
	x_train_flair_64x = x_train_flair_64x.reshape(x_train_flair_64x.shape[0], 1, Height, Width, Depth)
	x_train_T1_32x = x_train_T1_32x.reshape(x_train_T1_32x.shape[0], 1, Height, Width, Depth)
	x_train_T1_48x = x_train_T1_48x.reshape(x_train_T1_48x.shape[0], 1, Height, Width, Depth)       
	x_train_T1_64x = x_train_T1_64x.reshape(x_train_T1_64x.shape[0], 1, Height, Width, Depth)     
	
	y_test_flair_32x = y_test_flair_32x.reshape(y_test_flair_32x.shape[0], 1, Height, Width, Depth)
	y_test_flair_48x = y_test_flair_48x.reshape(y_test_flair_48x.shape[0], 1, Height, Width, Depth)    
	y_test_flair_64x = x_test_flair_64x.reshape(x_test_flair_64x.shape[0], 1, Height, Width, Depth)
	y_test_T1_32x = y_test_T1_32x.reshape(y_test_T1_32x.shape[0], 1, Height, Width, Depth)
	y_test_T1_48x = y_test_T1_48x.reshape(y_test_T1_48x.shape[0], 1, Height, Width, Depth)       
	y_test_T1_64x = y_test_T1_64x.reshape(y_test_T1_64x.shape[0], 1, Height, Width, Depth)    
	
	y_valid_flair_32x = y_valid_flair_32x.reshape(y_valid_flair_32x.shape[0], 1, Height, Width, Depth)
	y_valid_flair_48x = y_valid_flair_48x.reshape(y_valid_flair_48x.shape[0], 1, Height, Width, Depth)    
	y_valid_flair_64x = x_valid_flair_64x.reshape(x_valid_flair_64x.shape[0], 1, Height, Width, Depth)
	y_valid_T1_32x = y_valid_T1_32x.reshape(y_valid_T1_32x.shape[0], 1, Height, Width, Depth)
	y_valid_T1_48x = y_valid_T1_48x.reshape(y_valid_T1_48x.shape[0], 1, Height, Width, Depth)       
	y_valid_T1_64x = y_valid_T1_64x.reshape(y_valid_T1_64x.shape[0], 1, Height, Width, Depth)          

	# Converts a class vector (integers) to binary class matrix representation    
	tr_onehot_flair_32x = to_categorical(tr_lbl_flair_32x) 
	tr_onehot_flair_48x = to_categorical(tr_lbl_flair_48x)     
	tr_onehot_flair_64x = to_categorical(tr_lbl_flair_64x)  
	tr_onehot_T1_32x = to_categorical(tr_lbl_T1_32x) 
	tr_onehot_T1_48x = to_categorical(tr_lbl_T1_48x)     
	tr_onehot_T1_64x = to_categorical(tr_lbl_T1_64x)      
			
	ts_onehot_flair_32x = to_categorical(ts_lbl_flair_32x) 
	ts_onehot_flair_48x = to_categorical(ts_lbl_flair_48x)     
	ts_onehot_flair_64x = to_categorical(ts_lbl_flair_64x)  
	ts_onehot_T1_32x = to_categorical(ts_lbl_T1_32x) 
	ts_onehot_T1_48x = to_categorical(ts_lbl_T1_48x)     
	ts_onehot_T1_64x = to_categorical(ts_lbl_T1_64x)          
			
	val_onehot_flair_32x = to_categorical(val_lbl_flair_32x) 
	val_onehot_flair_48x = to_categorical(val_lbl_flair_48x)     
	val_onehot_flair_64x = to_categorical(val_lbl_flair_64x)  
	val_onehot_T1_32x = to_categorical(val_lbl_T1_32x) 
	val_onehot_T1_48x = to_categorical(val_lbl_T1_48x)     
	val_onehot_T1_64x = to_categorical(val_lbl_T1_64x) 

	return x_train_flair_32x, x_train_flair_48x, x_train_flair_64x, x_train_T1_32x, x_train_T1_48x, x_train_T1_64x, y_test_flair_32x,  y_test_flair_48x, y_test_flair_64x, y_test_T1_32x, y_test_T1_48x, y_test_T1_64x, y_valid_flair_32x, y_valid_flair_48x, y_valid_flair_64x, y_valid_T1_32x, y_valid_T1_48x, y_valid_T1_64x, tr_onehot_flair_32x, tr_onehot_flair_48x, tr_onehot_flair_64x, tr_onehot_T1_32x, tr_onehot_T1_48x, tr_onehot_T1_64x, ts_onehot_flair_32x, ts_onehot_flair_48x, ts_onehot_flair_64x, ts_onehot_T1_32x, ts_onehot_T1_48x, ts_onehot_T1_64x, val_onehot_flair_32x, val_onehot_flair_48x, val_onehot_flair_64x, val_onehot_T1_32x, val_onehot_T1_48x, val_onehot_T1_64x
