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

def get_index_2d (dim,n): #get caratesian coordinate
    j=n % dim
    j=j/1. #float( i)
    i=(n-j)/dim
    return (i,j)# pourquoi float?

def Balgovind(dim,L):
    sub_B=np.zeros((dim**2,dim**2))
    for i in range(dim**2):
        (a1,b1)=get_index_2d(dim,i)
        for j in range(dim**2):
            (a2,b2)=get_index_2d(dim,j) #reprends les donnees caracterisennes
            r=math.sqrt((a1-a2)**2+(b1-b2)**2)
            sub_B[i,j]=(1+r/L)*(math.exp(-r/L))
                                
                
    B1=np.concatenate((sub_B, np.zeros((dim**2,dim**2))), axis=1)
    B2=np.concatenate(( np.zeros((dim**2,dim**2)), sub_B),axis=1)
    B=np.concatenate((B1,B2), axis=0)# a changer construction matrice B
    return B[:dim**2,:dim**2]

#############################################################################
#load data

vor_test_u = np.load('DA_inverse_mapping/data/SW_vor_obs_u_R6_Hp02_random3.npy')
true_test_u = np.load('DA_inverse_mapping/data/SW_u_R6_Hp02.npy')
u = true_test_u[4000,:,:]

H = np.load('DA_inverse_mapping/data/H_obs.npy')

def marginal (x):
  #return np.square(x)
  return np.square(x)

def obs_field (x,H):
  x_m = marginal(x).reshape(-1,1)
  y = np.dot(H,x_m)
  return y

################################################################################
# synthetic observation
# here you can chose a different sample range or 

def stucture_obs(x_list,y_list):
  x_coord = []
  y_coord = []
  for i in range(len(x_list)-1):
    for j in range(len(y_list)-1):
      #x_coord.append(random.randint(x_list[i],x_list[i+1]))
      #y_coord.append(random.randint(y_list[j],y_list[j+1]))
      try:
        x_coord.append(random.randint(x_list[i],x_list[i]+sample_range))
        y_coord.append(random.randint(y_list[j],y_list[j]+sample_range))

      except:
        pass
  return x_coord,y_coord



x_list = [0,5,10,15,20,25,30,35,40,45,49]
y_list = [0,5,10,15,20,25,30,35,40,45,49]

x_coord,y_coord = stucture_obs(x_list,y_list)

obs_matrix = np.zeros((len(x_coord),2500))
line = 0
for i in range(len(x_coord)):
  sub_matrix = np.zeros((50,50))
  sub_matrix[x_coord[i],y_coord[i]]=1
  obs_matrix[line,:] = sub_matrix.ravel()
  line += 1


 x_m = obs_field (u,H).reshape(-1,1)
 y = np.dot(obs_matrix,x_m)

def obs_operator(x):
  x_m = obs_field (x,H).reshape(-1,1)
  return np.dot(obs_matrix,x_m)

u_b = u + 0.02*np.random.multivariate_normal(np.zeros(2500), Balgovind(50,5), 1).reshape(50,50)

#you can also load a simulated background state
#u_b = np.load('data/xb_DA_4000_final.npy')


#######################################################################################
#conventional variational data assimilation

case = adaoBuilder.New()
case.set( 'AlgorithmParameters', Algorithm='3DVAR',
         Parameters = {"Minimizer" : "LBFGSB","MaximumNumberOfSteps":1000,
                       "CostDecrementTolerance":1.e-6,
                       "StoreSupplementaryCalculations":["CostFunctionJ","CurrentState",
                        "SimulatedObservationAtOptimum",
                        "SimulatedObservationAtBackground",
                        "JacobianMatrixAtBackground",
                        "JacobianMatrixAtOptimum",
                        "KalmanGainAtOptimum",
                        "APosterioriCovariance"]
                       } )
case.set( 'Background',          Vector=u_b.ravel())
#case.set( 'BackgroundError',     ScalarSparseMatrix=1000.0 )
case.set( 'BackgroundError',     Matrix=100* Balgovind(50,1) )
case.set( 'Observation',         Vector=y )
case.set( 'ObservationError',    ScalarSparseMatrix=1.0 )
case.set( 'ObservationOperator', OneFunction = obs_operator)
case.set( 'Observer',            Variable="Analysis", Template="ValuePrinter" )

case.setObserver(Variable="CostFunctionJ",Template="ValuePrinter")
case.setObserver(Variable="CostFunctionJo",Template="ValuePrinter")
case.setObserver(Variable="CostFunctionJb",Template="ValuePrinter")
case.execute()

x_a = case.get("Analysis")[-1]
J = case.get("CostFunctionJ")
Jo = case.get("CostFunctionJo")
Jb = case.get("CostFunctionJb")

#########################################################################################
# plot the evolution of the objective function in DA

plt.plot(list(J),label ='J')
plt.plot(list(Jo),'r',label ='Jo')
plt.plot(list(Jb),'g',label ='Jb')
plt.yscale('log')
plt.xlabel('Iterations',fontsize=16)
plt.legend(fontsize=16)

# plot assimilated state and error
im = plt.imshow(x_a.reshape(50,50)-u.reshape(50,50),vmin=0,vmax=0.02,cmap='jet')
plt.colorbar(im)
plt.axis('off')

im = plt.imshow(x_a.reshape(50,50),vmin=-0.03,vmax=0.03)
plt.colorbar(im)
plt.axis('off')

print('background error',np.linalg.norm(u_b.reshape(50,50) - u)/np.linalg.norm(u))
print('analysis error',np.linalg.norm(x_a.reshape(50,50) - u)/np.linalg.norm(u))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
print('background ssim',ssim(u_b.reshape(50,50),u))
print('analysis ssim',ssim(x_a.reshape(50,50),u))

############################################################################################
#load trained VCNN inverse operator
model =  keras.models.load_model('model/SW_VC_u_random3_final.h5')


#construct voronoi for VIVID

Ni = 100
N = 50
Ni_portion = np.ones((Ni, 2))
Ni_portion[:,1] = np.zeros(Ni)


Xi, Yi = np.array(x_coord), np.array(y_coord)
Pi = np.zeros((len(x_coord),2))
Pi[:,0] = Xi
Pi[:,1] = Yi

#Zi = SW.u[Xi.astype(int),50-1-Yi.astype(int)]
x_m = x_m.reshape(50,50)
Zi = x_m[Xi.astype(int),50-1-Yi.astype(int)]

x = np.linspace(0., N, N)
y = np.linspace(0., N, N)[::-1]
X, Y = np.meshgrid(x, y)
P = np.array([X.flatten(), Y.flatten() ]).transpose()


from scipy.interpolate import griddata
Z_nearest = griddata(Pi, Zi, P, method = "nearest").reshape([N, N])
#plt.contourf(Y, X, Z_nearest, 50,vmax=np.max(SW.u),vmin=np.min(SW.u))
#im = plt.imshow(Z_nearest.T,vmax=np.max(SW.u),vmin=np.min(SW.u))
im = plt.imshow(Z_nearest.T)
#plt.plot(Xi, Yi, "or", label = "Data")
#plt.colorbar()
plt.scatter(50-1-Yi,Xi,color='r')
plt.legend()
plt.grid()
plt.axis('off')
plt.show()

x_m = obs_field (u,H).reshape(-1,1)
y = np.dot(obs_matrix,x_m)
inversed_u = model.predict(Z_nearest.T.reshape(1,50,50)).ravel()

##########################################################################################
# VIVID assimilation

y_combined = np.array(list(inversed_u) + list(y))
def obs_operator(x):
  x_m = obs_field (x,H).reshape(-1,1)
  output = list(x)+list(np.dot(obs_matrix,x_m).ravel())
  return output

R_diag = list(100*np.ones(2500))+list(1*np.ones(100))



case = adaoBuilder.New()
case.set( 'AlgorithmParameters', Algorithm='3DVAR',
         Parameters = {"Minimizer" : "LBFGSB","MaximumNumberOfSteps":1000,
                       "CostDecrementTolerance":1.e-6,
                       "StoreSupplementaryCalculations":["CostFunctionJ","CurrentState",
                        "SimulatedObservationAtOptimum",
                        "SimulatedObservationAtBackground",
                        "JacobianMatrixAtBackground",
                        "JacobianMatrixAtOptimum",
                        "KalmanGainAtOptimum",
                        "APosterioriCovariance"]
                       } )
case.set( 'Background',          Vector=u_b.ravel())
#case.set( 'BackgroundError',     ScalarSparseMatrix=1000.0 )
case.set( 'BackgroundError',     Matrix=1000* Balgovind(50,5) )
case.set( 'Observation',         Vector=y_combined )
case.set( 'ObservationError',     DiagonalSparseMatrix = R_diag)
#case.set( 'ObservationError',     Matrix = R)
case.set( 'ObservationOperator', OneFunction = obs_operator)
case.set( 'Observer',            Variable="Analysis", Template="ValuePrinter" )

case.setObserver(Variable="CostFunctionJ",Template="ValuePrinter")
case.setObserver(Variable="CostFunctionJo",Template="ValuePrinter")
case.setObserver(Variable="CostFunctionJb",Template="ValuePrinter")
case.execute()

x_a = case.get("Analysis")[-1]
J = case.get("CostFunctionJ")
Jo = case.get("CostFunctionJo")
Jb = case.get("CostFunctionJb")

###################################################################################################

plt.plot(list(J),label ='J')
plt.plot(list(Jo),'r',label ='Jo+Jp')
plt.plot(list(Jb),'g',label ='Jb')
plt.yscale('log')
plt.xlabel('Iterations',fontsize=16)
plt.legend(fontsize=16)

im = plt.imshow(np.abs(inversed_u.reshape(50,50)-u),vmin=0,vmax=0.03,cmap='jet')
plt.colorbar(im)
plt.axis('off')

im = plt.imshow(np.abs(x_a.reshape(50,50)-u),vmin=0,vmax=0.03,cmap='jet')
plt.colorbar(im)
plt.axis('off')

print('background error',np.linalg.norm(u_b.reshape(50,50) - u)/np.linalg.norm(u))
print('inverse error',np.linalg.norm(inversed_u.reshape(50,50) - u)/np.linalg.norm(u))
print('analysis error',np.linalg.norm(x_a.reshape(50,50) - u)/np.linalg.norm(u))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
print('background ssim',ssim(u_b.reshape(50,50),u))
print('inverse ssim',ssim(inversed_u.reshape(50,50),u))
print('analysis ssim',ssim(x_a.reshape(50,50),u))
