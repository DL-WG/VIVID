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

#build the CNN model

input_img = Input(shape=(50,50,1))
x = Convolution2D(48, (8,8),activation='relu', padding='same')(input_img)
x = Convolution2D(48, (8,8),activation='relu', padding='same')(x)
x = Convolution2D(48, (8,8),activation='relu', padding='same')(x)
x = Convolution2D(48, (8,8),activation='relu', padding='same')(x)
x = Convolution2D(48, (8,8),activation='relu', padding='same')(x)
#x = Convolution2D(48, (8,8),activation='relu', padding='same')(x)
x = Convolution2D(48, (8,8),activation='relu', padding='same')(x)
x_final = Convolution2D(1, (8,8), padding='same')(x)
model = Model(input_img, x_final)
model.compile(optimizer='adam', loss='mse')

K.set_value(model.optimizer.learning_rate, 0.0001)
history = model.fit(vor_train_u, true_train_u,epochs=20,validation_split=0.05, batch_size=64,shuffle=True,verbose = 2)
model.save('drive/MyDrive/DA_inverse_mapping/model/SW_VC_u_random3_final.h5')

##############################################################################
# model test

vor_test_u = np.load('drive/MyDrive/DA_inverse_mapping/data/SW_vor_obs_u_R6_Hp02_random3.npy')
true_test_u = np.load('drive/MyDrive/DA_inverse_mapping/data/SW_u_R6_Hp02.npy')

test_output_u = model.predict(vor_test_u)
train_output_u = model.predict(vor_train_u)

snap_index = 1000

plt.imshow(train_output_u[snap_index,:,:].reshape(50,50))
plt.show()
plt.close()

plt.imshow(test_output_u[snap_index,:,:].reshape(50,50))
plt.show()
plt.close()

###############################################################################
# evaluating the difference of different train/test dataset

R4_Hp01 = np.load('DA_inverse_mapping/data/SW_u_R4_Hp01.npy')
R4_Hp015 = np.load('DA_inverse_mapping/data/SW_u_R4_Hp015.npy')
R5_Hp01 = np.load('DA_inverse_mapping/data/SW_u_R5_Hp01.npy')
R5_Hp015 = np.load('DA_inverse_mapping/data/SW_u_R5_Hp015.npy')
R6_Hp02 = np.load('DA_inverse_mapping/data/SW_u_R6_Hp02.npy')


from skimage.metrics import structural_similarity
def RMSE(A,B):
  return np.linalg.norm(A-B)/np.linalg.norm(A)

R4_Hp01_RMSE = []
R4_Hp015_RMSE = []
R5_Hp01_RMSE = []
R5_Hp015_RMSE = []

R4_Hp01_SSIM = []
R4_Hp015_SSIM = []
R5_Hp01_SSIM = []
R5_Hp015_SSIM = []

for i in range(10000):
  R4_Hp01_RMSE.append(RMSE(R6_Hp02[i,:,:],R4_Hp01[i,:,:]))
  R4_Hp015_RMSE.append(RMSE(R6_Hp02[i,:,:],R4_Hp015[i,:,:]))
  R5_Hp01_RMSE.append(RMSE(R6_Hp02[i,:,:],R5_Hp01[i,:,:]))
  R5_Hp015_RMSE.append(RMSE(R6_Hp02[i,:,:],R5_Hp015[i,:,:]))

  R4_Hp01_SSIM.append(structural_similarity(R6_Hp02[i,:,:],R4_Hp01[i,:,:], multichannel=True))
  R4_Hp015_SSIM.append(structural_similarity(R6_Hp02[i,:,:],R4_Hp015[i,:,:], multichannel=True))
  R5_Hp01_SSIM.append(structural_similarity(R6_Hp02[i,:,:],R5_Hp01[i,:,:], multichannel=True))
  R5_Hp015_SSIM.append(structural_similarity(R6_Hp02[i,:,:],R5_Hp015[i,:,:], multichannel=True))


from matplotlib.ticker import FuncFormatter

plt.plot(R4_Hp01_RMSE,label='$r_s=4, h_p=0.1$')
plt.plot(R4_Hp015_RMSE,label='$r_s=4, h_p=0.15$')
plt.plot(R5_Hp01_RMSE,label='$r_s=5, h_p=0.1$')
plt.plot(R5_Hp015_RMSE,label='$r_s=5, h_p=0.15$')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.xlabel('time step',fontsize = 16)
plt.ylabel('R-RMSE',fontsize = 16)
plt.legend()


plt.plot(R4_Hp01_SSIM,label='$r_s=4, h_p=0.1$')
plt.plot(R4_Hp015_SSIM,label='$r_s=4, h_p=0.15$')
plt.plot(R5_Hp01_SSIM,label='$r_s=5, h_p=0.1$')
plt.plot(R5_Hp015_SSIM,label='$r_s=5, h_p=0.15$')
#plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.xlabel('time step',fontsize = 16)
plt.ylabel('SSIM',fontsize = 16)
plt.legend()
