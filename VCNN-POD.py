import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import animation as animation
from PIL import Image
import matplotlib as mpl
import time
from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Cropping2D, AveragePooling2D,Dense,Flatten,Reshape,Dropout
from tensorflow.python.keras.models import Model,Sequential
import tensorflow as tf
import keras
from keras.layers import LeakyReLU
from keras import backend as K
from keras import optimizers

#########################################################################
#loading training datasets

vor_train_u = np.load('DA_inverse_mapping/data/SW_vor_obs_u_R4_Hp01_random3.npy')
vor_train_u = np.concatenate((vor_train_u, np.load('DA_inverse_mapping/data/SW_vor_obs_u_R4_Hp015_random3.npy')),axis = 0)
vor_train_u = np.concatenate((vor_train_u, np.load('DA_inverse_mapping/data/SW_vor_obs_u_R5_Hp01_random3.npy')),axis = 0)
vor_train_u = np.concatenate((vor_train_u, np.load('DA_inverse_mapping/data/SW_vor_obs_u_R5_Hp015_random3.npy')),axis = 0)

true_train_u = np.load('DA_inverse_mapping/data/SW_u_R4_Hp01.npy')
true_train_u = np.concatenate((true_train_u,np.load('DA_inverse_mapping/data/SW_u_R4_Hp015.npy')),axis=0)
true_train_u = np.concatenate((true_train_u,np.load('DA_inverse_mapping/data/SW_u_R5_Hp01.npy')),axis=0)
true_train_u = np.concatenate((true_train_u,np.load('DA_inverse_mapping/data/SW_u_R5_Hp015.npy')),axis=0)

true_train_u = true_train_u.astype('float32')
vor_train_u = vor_train_u.astype('float32')

##########################################################################

#compute the POD

u_pod, s_pod, v_pod = np.linalg.svd(true_train_u.T, full_matrices=False)
plt.plot(s_pod[:200],label='singular value')
plt.xlabel('truncation parameter',fontsize = 15)
plt.legend(fontsize=16)
plt.show()
plt.close()

#evaluate the POD accuracy

u_pod_q = u_pod[:,:100]
field_compress_PCA = np.dot(u_pod_q.T,true_test_u.T)
field_reconstruct_PCA = np.dot(u_pod_q,field_compress_PCA).T.reshape(-1,50,50)

print(np.linalg.norm(field_reconstruct_PCA-true_test_u.reshape(-1,50,50))/np.linalg.norm(true_test_u))

#####################################################################################
# train the CNN from tessellated observation space to the reduced state space

input_img = Input(shape=(50,50, 1))
 
x = Convolution2D(16, (8, 8), activation='relu', padding='same')(input_img)
x = Convolution2D(16, (8, 8), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(16, (4, 4), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(16, (4, 4), activation='relu', padding='same')(x)
flat = Flatten()(x)
encoded = Dense(100)(flat)
encoder = Model(input_img, encoded)

encoder.compile(optimizer='adam', loss='mse')

###################################################################################### 

history = encoder.fit(vor_train_u.reshape(-1,50,50,1), field_compress_PCA.T.reshape(-1,100,1), epochs=400, validation_split=0.1,batch_size=64,shuffle=True,verbose=2) 
encoder.save('model/SW_VC_u_random3_POD_final.h5')

#######################################################################################
#Evaluate VCNN-POD

predicted_PCA = encoder.predict(vor_test_u).T
field_reconstruct_PCA = np.dot(u_pod_q,predicted_PCA).T.reshape(-1,50,50)

sample_index = 5000
plt.imshow(field_reconstruct_PCA[sample_index,:,:])