# In this cript we construct the training data including both the state field (velocity field u) and the synthetic observation data

from shallow_water import *

#############################################################
#parameters
# The radius R and the cylindre water level hp
R = 4
hp=0.1

#grid points of observations

x_list = [0,5,10,15,20,25,30,35,40,45,49]
y_list = [0,5,10,15,20,25,30,35,40,45,49]

#sample range of sensor placements around the grid points
sample_range = 3 
#############################################################


#define the distance metric
def dist_2d (x,y):
  return (np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2))

#construct the observation function
H = np.zeros((2500,2500))
line = 0
for i in range(50):
  for j in range(50):
    pt1 = (i,j)
    sub_matrix = np.zeros((50,50))
    for q in range(50):
      for t in range(50):
        pt2 = (q,t)
        if dist_2d(pt1,pt2) <=2.9: 
          sub_matrix[q,t] += 0.5
          #sub_matrix[q,t] +=random.uniform(0.4, 0.6)
        if dist_2d(pt1,pt2) <=1.5: 
          sub_matrix[q,t] += 1
          #sub_matrix[q,t] +=random.uniform(1.2, 1.8)
    H[line,:] = sub_matrix.ravel()
    line += 1

def marginal (x):
  return np.square(x)

def obs_field (x,H):
  x_m = marginal(x).reshape(-1,1)
  y = np.dot(H,x_m)
  return y


obs_matrix = np.zeros((len(x_coord),2500))
line = 0
for i in range(len(x_coord)):
  sub_matrix = np.zeros((50,50))
  sub_matrix[x_coord[i],y_coord[i]]=1
  obs_matrix[line,:] = sub_matrix.ravel()
  line += 1

x_m = obs_field (SW.u,H).reshape(-1,1)
y = np.dot(obs_matrix,x_m)

#####################################################################
#define the random sensor placement

Ni = 100
N = 50
Ni_portion = np.ones((Ni, 2))
Ni_portion[:,1] = np.zeros(Ni)

def stucture_obs(x_list,y_list):
  x_coord = []
  y_coord = []
  for i in range(len(x_list)-1):
    for j in range(len(y_list)-1):
      #x_coord.append(random.randint(x_list[i],x_list[i+1]))
      #y_coord.append(random.randint(y_list[j],y_list[j+1]))
      try:
        x_coord.append(x_list[i]+sample_range)
        y_coord.append(y_list[j]+sample_range)
      except:
        pass
  return x_coord,y_coord

#######################################################################
#generating training and test datasets
iteration_times=10000


SW_h = np.zeros((iteration_times,50,50))
SW_u = np.zeros((iteration_times,50,50))
SW_v = np.zeros((iteration_times,50,50))

SW_vor_u = np.zeros((iteration_times,50,50))

#iteration_times2=1
#SW = shallow(px=40,py=30)
dim = 50
u=np.zeros((dim,dim))
v=np.ones((dim,dim))
    

h=1. * ones((dim,dim))
    

px = np.round(dim/2)
py = np.round(dim/2)

for x in range(dim):
    for y in range(dim):
        if (x-px)**2 + (y-py)**2 <= R**2:
          h[x,y] = 1+hp
    
SW = shallow_dynamique(u=u,v=v,h=h)



    
##############################################################################################
# chose a point (x,y) to check the evolution

for i in range(iteration_times):


    SW.evolve()

    if i % 100 == 0:
      
SW_h = np.zeros((iteration_times,50,50))
SW_u = np.zeros((iteration_times,50,50))
SW_v = np.zeros((iteration_times,50,50))

SW_vor_u = np.zeros((iteration_times,50,50))

#iteration_times2=1
#SW = shallow(px=40,py=30)
dim = 50
u=np.zeros((dim,dim))
v=np.ones((dim,dim))

#h = np.loadtxt('dataMC_500_004/h_initial_500.txt')

h=1. * ones((dim,dim))

px = np.round(dim/2)
py = np.round(dim/2)

for x in range(dim):
  for y in range(dim):
    if (x-px)**2 + (y-py)**2 <= R**2:
      h[x,y] = 1 + Hp

SW = shallow_dynamique(u=u,v=v,h=h)

plt.imshow(SW.h)
plt.show()
plt.close()
# chose a point (x,y) to check the evolution


#SW.plot()

##############################################################################################
# chose a point (x,y) to check the evolution

for i in range(iteration_times):


    SW.evolve()

    if i % 1 == 0:
      
        SW_h[i,:,:] = SW.h
        SW_u[i,:,:] = SW.u
        SW_v[i,:,:] = SW.v


        x_coord,y_coord = stucture_obs(x_list,y_list)

        print ('time %f'%SW.time)
##            SW.fig[-1].savefig('sw_%.3d.png'% i)
        #plt.imshow(SW.u)
        #plt.show()
        #plt.close()
        obs_field_t = obs_field(SW.u,H).reshape(50,50)
#############voronoi###############################################
        Xi, Yi = np.array(x_coord), np.array(y_coord)
        Pi = np.zeros((len(x_coord),2))
        Pi[:,0] = Xi
        Pi[:,1] = Yi

        Zi = obs_field_t[Xi.astype(int),50-1-Yi.astype(int)]

        x = np.linspace(0., N, N)
        y = np.linspace(0., N, N)[::-1]
        X, Y = np.meshgrid(x, y)
        P = np.array([X.flatten(), Y.flatten() ]).transpose()

        Z_nearest = griddata(Pi, Zi, P, method = "nearest").reshape([N, N])

        SW_vor_u[i,:,:] = Z_nearest.T

np.save('DA_inverse_mapping/data/SW_vor_obs.npy',SW_vor_u)
np.save('DA_inverse_mapping/data/SW_u.npy',SW_u)