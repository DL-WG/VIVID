# VIVID


## About the project
We have developed a new method for variational data assimilation called VIVID, which incorporates a deep learning inverse operator into the assimilation objective function. This technique utilizes Voronoi-tessellation and convolutional neural networks to effectively handle sparse, unstructured, and time-varying sensor data. By incorporating the DL inverse operator, VIVID establishes a direct connection between observation and state space, which minimizes the number of steps required for data assimilation. The numerical experiments are made with a 2D shallow code provided in this reporsitory. 

## Getting Started

*   Programming language: Python (3.5 or higher)


### Software requirement

| Package Requirement                        |
|--------------------------------------------|
| os                                         |
| numpy                                      |
| pandas                                     |
| math                                       |
| matplotlib                                 |
| ADAO (9.10.0)                              |
| Tensorflow (2.3.0 or higher)               |
| Keras (2.4.0 or higher)                    |

## Dataset 
The experimental data are generated by the shallow_water.py file

4 simulations (each of 10000 time steps) are used as training set while one simulation is used to test the performance of the proposed algorithm

Training data parameters, (hp, rw): (0.1, 4), (0.15, 4), (0.1, 5), (0.15, 5)
Test data parameters, (hp, rw): (0.2, 6)
 
 
## VCNN preprocessing and training
The preprocessing of the observations through Voronoi tessellation is presented in voronoi_preprocessing.py
The traning of VCNN is presented in VCNN_training.py


