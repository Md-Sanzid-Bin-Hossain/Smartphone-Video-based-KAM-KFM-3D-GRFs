

import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import numpy
import statistics
from numpy import loadtxt
import matplotlib.pyplot as plt
import pandas
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statistics import stdev
import math
import h5py

import numpy as np
import time

from scipy.signal import butter,filtfilt
import sys
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
import pandas
import matplotlib.pyplot as plt

# from tsf.model import TransformerForecaster


# from tensorflow.keras.utils import np_utils
import itertools
###  Library for attention layers
import pandas as pd
import os
import numpy as np
#from tqdm import tqdm # Processing time measurement
from sklearn.model_selection import train_test_split

import statistics
import gc
import torch.nn.init as init

############################################################################################################################################################################
############################################################################################################################################################################

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.utils.weight_norm as weight_norm


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torchsummary import summary
from torch.nn.parameter import Parameter


import torch.optim as optim


from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler

### Augmentation Techniques ###


def augment_data(input_data, noise_factor):
    """
    Augments input data by adding random noise to each value.

    Args:
        input_data (torch.Tensor): Input data to be augmented.
        noise_factor (float): Scaling factor for the random noise.

    Returns:
        torch.Tensor: Augmented data with added noise.
    """
    # Generate random noise of the same shape as input_data
    noise = torch.randn_like(input_data) * noise_factor

    # Add the noise to the input_data
    augmented_data = input_data + noise

    return np.array(augmented_data)
    
    
    
# Data loader

def extract_vel_acc(V):

  velocity_all = []
  acceleration_all = []

  for i in range(44):
      velocity, acceleration = calculate_velocity_acceleration(V[:,i])

      velocity_all.append(velocity)
      acceleration_all.append(velocity)

  return np.transpose(velocity_all), np.transpose(acceleration_all)

def calculate_velocity_acceleration(position_data):
    n = len(position_data)

    # Calculate velocity
    velocity = []
    for i in range(n):
        if i == 0:
            vel = 0.0  # Set initial velocity as 0
        else:
            displacement = position_data[i] - position_data[i-1]
            time_interval = 0.01  # Assuming time intervals are uniform (e.g., 1 second)
            vel = displacement / time_interval
        velocity.append(vel)

    # Calculate acceleration
    acceleration = []
    for i in range(n):
        if i < 2 or i > n-2:
            accel = 0.0  # Set acceleration as 0 for the first and last points
        else:
            velocity_change = velocity[i] - velocity[i-1]
            accel = velocity_change / time_interval
        acceleration.append(accel)

    return np.array(velocity), np.array(acceleration)

if __name__ == '__main__':
    with h5py.File('/home/sanzidpr/all_17_subjects.h5', 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])

def data_extraction(A):
  for k in range(len(A)):
    zero_index_1=np.all(A[k:k+1,:,:] == 0, axis=0)
    zero_index = np.multiply(zero_index_1, 1)
    zero_index=np.array(zero_index)

    for i in range(len(zero_index)):
      if (sum(zero_index[i])==256):
        index=i
        break;

    # print(index)
### Taking only the stance phase of the gait
###################################################################################################################################################
    B=A[k:k+1,0:index,:]  ### Taking only the stance phase of the gait
    C_1=B.reshape((B.shape[0]*B.shape[1],B.shape[2]))
    if (k==0):
      C=C_1
    else:
      C=np.append(C,C_1,axis=0)

  index_24 = data_fields.index('body weight')
  index_25 = data_fields.index('body height')

  BW=(C[0:1, index_24]*9.8)
  BWH=(C[0:1, index_24]*9.8)*C[:, index_25]

  V=C[:,110:154]
  V=V.reshape(V.shape[0],11,4)

  V=(V-V[:,2:3,:])
  # V=(V-V[:,2:3,:])/(C[0:1, index_25]*1000)
  V=V.reshape(-1,44)

  velocity_all, acceleration_all = extract_vel_acc(V)
  
  V=V/C[0:1, index_25]


      ### IMUs- Chest, Waist, Right Foot, Right shank, Right thigh, Left Foot, Left shank, Left thigh, 2D-body coordinate
    ### 0:48- IMU, 48:92-2D body coordinate, 92:136 -2D velocity, 136:180 -2D acceleration, 180:185-- Target

  D=np.hstack((C[:,71:77],C[:,58:64],C[:,19:25],C[:,32:38],C[:,45:51],C[:,6:12],C[:,84:90],C[:,97:103],V,velocity_all, acceleration_all, C[:,3:5],-C[:, 154:155]/BW,
              -C[:, 156:157]/BW,-C[:, 155:156]/BW))



  return D

  # index_21 = data_fields.index('plate_2_force_x')
  # print(index_21)

# print(np.array(data_fields))

data_subject_01 = data_all_sub['subject_01']
subject_1=data_extraction(data_subject_01)

print(subject_1.shape)

data_subject_01 = data_all_sub['subject_01']
data_subject_02 = data_all_sub['subject_02']
data_subject_03 = data_all_sub['subject_03']
data_subject_04 = data_all_sub['subject_04']
data_subject_05 = data_all_sub['subject_05']
data_subject_06 = data_all_sub['subject_06']
data_subject_07 = data_all_sub['subject_07']
data_subject_08 = data_all_sub['subject_08']
data_subject_09 = data_all_sub['subject_09']
data_subject_10 = data_all_sub['subject_10']
data_subject_11 = data_all_sub['subject_11']
data_subject_12 = data_all_sub['subject_12']
data_subject_13 = data_all_sub['subject_13']
data_subject_14 = data_all_sub['subject_14']
data_subject_15 = data_all_sub['subject_15']
data_subject_16 = data_all_sub['subject_16']
data_subject_17 = data_all_sub['subject_17']


subject_1=data_extraction(data_subject_01)
subject_2=data_extraction(data_subject_02)
subject_3=data_extraction(data_subject_03)
subject_4=data_extraction(data_subject_04)
subject_5=data_extraction(data_subject_05)
subject_6=data_extraction(data_subject_06)
subject_7=data_extraction(data_subject_07)
subject_8=data_extraction(data_subject_08)
subject_9=data_extraction(data_subject_09)
subject_10=data_extraction(data_subject_10)
subject_11=data_extraction(data_subject_11)
subject_12=data_extraction(data_subject_12)
subject_13=data_extraction(data_subject_13)
subject_14=data_extraction(data_subject_14)
subject_15=data_extraction(data_subject_15)
subject_16=data_extraction(data_subject_16)
subject_17=data_extraction(data_subject_17)


subject_1.shape


##########################################################################################################################################################
##########################################################################################################################################################

adjacency_matrix = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])

adjacency_matrix=torch.from_numpy(adjacency_matrix.astype(np.float32))


def graph_based_augmentation(joints, adjacency_matrix, max_rotation, max_translation):
    num_joints = joints.shape[0]

    # Apply random rotations
    theta = torch.randn(num_joints, 1) * (max_rotation * np.pi / 180)  # Convert to radians
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    rotated_joints = torch.zeros_like(joints)
    for i in range(num_joints):
        neighbors = torch.nonzero(adjacency_matrix[i]).squeeze(1)
        rotated_joints[i] = cos_theta[i] * joints[i] + torch.sum(sin_theta[i] * joints[neighbors], dim=0)

    # Apply random translations
    translation = torch.randn(2) * max_translation

    augmented_joints = rotated_joints + translation

    return augmented_joints


def augmentation_all(V):

    joints=V
    augmented_joints_all=[]

    for i in range(len(joints)):

      joint_1=joints[i,:,0:2].squeeze(0)
      augmented_joints_1 = graph_based_augmentation(joint_1, adjacency_matrix, max_rotation=2.0, max_translation=1.0)

      joint_2=joints[i,:,2:4].squeeze(0)
      augmented_joints_2 = graph_based_augmentation(joint_2, adjacency_matrix, max_rotation=2.0, max_translation=1.0)

      augmented_joints_1=augmented_joints_1.unsqueeze(0)
      augmented_joints_2=augmented_joints_2.unsqueeze(0)

      augmented_joints=torch.cat((augmented_joints_1,augmented_joints_2),dim=-1)
      augmented_joints_all.append(augmented_joints)

    augmented_joints_all = torch.stack(augmented_joints_all, dim=0)
    augmented_joints_all=augmented_joints_all.unsqueeze(1)

    return augmented_joints_all


def IMU_augmentation(IMU_data, angle):

  angle_rad = np.radians(angle)
  rotation_matrix_1 = np.array([[1, 0, 0],
                                    [0, np.cos(angle_rad), -np.sin(angle_rad)],
                                    [0, np.sin(angle_rad), np.cos(angle_rad)]])

  rotation_matrix_2 = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                                    [0, 1, 0],
                                    [-np.sin(angle_rad), 0, np.cos(angle_rad)]])

  rotation_matrix_3 = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                    [np.sin(angle_rad), np.cos(angle_rad), 0],
                                    [0, 0, 1]])
  
  IMU_data=np.concatenate((np.matmul(IMU_data[:,:,0:3],rotation_matrix_1),np.matmul(IMU_data[:,:,3:6],rotation_matrix_1)),axis=-1)
  IMU_data=np.concatenate((np.matmul(IMU_data[:,:,0:3],rotation_matrix_2),np.matmul(IMU_data[:,:,3:6],rotation_matrix_2)),axis=-1)
  IMU_data=np.concatenate((np.matmul(IMU_data[:,:,0:3],rotation_matrix_3),np.matmul(IMU_data[:,:,3:6],rotation_matrix_3)),axis=-1)

  return IMU_data
  

def data_extraction_aug(A):
  for k in range(len(A)):
    zero_index_1=np.all(A[k:k+1,:,:] == 0, axis=0)
    zero_index = np.multiply(zero_index_1, 1)
    zero_index=np.array(zero_index)

    for i in range(len(zero_index)):
      if (sum(zero_index[i])==256):
        index=i
        break;

    # print(index)
### Ta2Dg only the stance phase of the gait
###################################################################################################################################################
    B=A[k:k+1,0:index,:]  ### Ta2Dg only the stance phase of the gait
    C_1=B.reshape((B.shape[0]*B.shape[1],B.shape[2]))
    if (k==0):
      C=C_1
    else:
      C=np.append(C,C_1,axis=0)

  index_24 = data_fields.index('body weight')
  index_25 = data_fields.index('body height')

  BW=(C[0:1, index_24]*9.8)
  BWH=(C[0:1, index_24]*9.8)*C[:, index_25]

  V=C[:,110:154]
  V=V.reshape(V.shape[0],11,4)

  V=(V-V[:,2:3,:])
  
  V=augment_data(torch.from_numpy(V), 0.50)
  
  
  
  V=V.reshape(-1,44)
  velocity_all, acceleration_all = extract_vel_acc(V)

  V=V/C[0:1, index_25]

  IMU_data=np.hstack((C[:,71:77],C[:,58:64],C[:,19:25],C[:,32:38],C[:,45:51],C[:,6:12],C[:,84:90],C[:,97:103]))
#  IMU_data=IMU_data.reshape(IMU_data.shape[0],8,6)
#
#  IMU_data=IMU_augmentation(IMU_data, 0.5)
#  IMU_data=IMU_data.reshape(IMU_data.shape[0],8*6)

         
  ### IMUs- Chest, Waist, Right Foot, Right shank, Right thigh, Left Foot, Left shank, Left thigh, 2D-body coordinate
  ### 0:48- IMU, 48:92-2D body coordinate, 92:136 -2D velocity, 136:180 -2D acceleration, 180:185-- Target

  D=np.hstack((IMU_data, V, velocity_all, acceleration_all, C[:,3:5],-C[:, 154:155]/BW,
              -C[:, 156:157]/BW,-C[:, 155:156]/BW))

  return D


##########################################################################################################################################################
##########################################################################################################################################################

#
###
subject_1_aug=data_extraction_aug(data_subject_01)
subject_2_aug=data_extraction_aug(data_subject_02)
subject_3_aug=data_extraction_aug(data_subject_03)
subject_4_aug=data_extraction_aug(data_subject_04)
subject_5_aug=data_extraction_aug(data_subject_05)
subject_6_aug=data_extraction_aug(data_subject_06)
subject_7_aug=data_extraction_aug(data_subject_07)
subject_8_aug=data_extraction_aug(data_subject_08)
subject_9_aug=data_extraction_aug(data_subject_09)
subject_10_aug=data_extraction_aug(data_subject_10)
subject_11_aug=data_extraction_aug(data_subject_11)
subject_12_aug=data_extraction_aug(data_subject_12)
subject_13_aug=data_extraction_aug(data_subject_13)
subject_14_aug=data_extraction_aug(data_subject_14)
subject_15_aug=data_extraction_aug(data_subject_15)
subject_16_aug=data_extraction_aug(data_subject_16)
subject_17_aug=data_extraction_aug(data_subject_17)



########################################################################################################################################################################################
########################################################################################################################################################################################



# Data processing


##########################################################################################################################################################################################





main_dir = "/home/sanzidpr/Journal_4/Dataset_B_model_results/Subject17"
#os.mkdir(main_dir) 
path="/home/sanzidpr/Journal_4/Dataset_B_model_results/Subject17/"
subject='Subject_17'

train_dataset=np.concatenate((subject_1,subject_2,subject_3,subject_4,subject_5,subject_6,subject_7,subject_8,subject_9,subject_10,
                              subject_11,subject_12,subject_13,subject_14,subject_15,subject_16),axis=0)
                              
                                                        
                              
train_dataset_aug_1=np.concatenate((subject_1_aug,subject_2_aug,subject_3_aug,subject_4_aug,subject_5_aug,subject_6_aug,subject_7_aug,subject_8_aug,subject_9_aug,subject_10_aug,
                              subject_11_aug,subject_12_aug,subject_13_aug,subject_14_aug,subject_15_aug,subject_16_aug),axis=0)                              
                              
                              
                              
train_dataset_aug=np.concatenate((train_dataset,train_dataset_aug_1),axis=0)                              

test_dataset=subject_17




############################################################################################################################################


encoder='lstm_gcn'

# Data processing



# x_train_1=train_dataset[:,0:18]
# x_train_2=train_dataset[:,23:67]

# x_train=np.concatenate((x_train_1,x_train_2),axis=1)

x_train=train_dataset[:,0:180]


train_X_1_1=x_train

# # Test features #
# x_test_1=test_dataset[:,0:18]
# x_test_2=test_dataset[:,23:67]

# x_test=np.concatenate((x_test_1,x_test_2),axis=1)

x_test=test_dataset[:,0:180]

test_X_1_1=x_test

m1=180
m2=185


  ### Label ###

train_y_1_1=train_dataset[:,m1:m2]
test_y_1_1=test_dataset[:,m1:m2]

train_dataset_1=np.concatenate((train_X_1_1,train_y_1_1),axis=1)
test_dataset_1=np.concatenate((test_X_1_1,test_y_1_1),axis=1)

train_dataset_1=pd.DataFrame(train_dataset_1)
test_dataset_1=pd.DataFrame(test_dataset_1)

train_dataset_1.dropna(axis=0,inplace=True)
test_dataset_1.dropna(axis=0,inplace=True)

train_dataset_1=np.array(train_dataset_1)
test_dataset_1=np.array(test_dataset_1)

train_dataset_sum = np. sum(train_dataset_1)
array_has_nan = np. isinf(train_dataset_1[:,48:180])

print(array_has_nan)

print(train_dataset_1.shape)



train_X_1=train_dataset_1[:,0:m1]
test_X_1=test_dataset_1[:,0:m1]

train_y_1=train_dataset_1[:,m1:m2]
test_y_1=test_dataset_1[:,m1:m2]



L1=len(train_X_1)
L2=len(test_X_1)


w=50



a1=L1//w
b1=L1%w

a2=L2//w
b2=L2%w

# a3=L3//w
# b3=L3%w

     #### Features ####
train_X_2=train_X_1[L1-w+b1:L1,:]
test_X_2=test_X_1[L2-w+b2:L2,:]
# validation_X_2=validation_X_1[L3-w+b3:L3,:]


    #### Output ####

train_y_2=train_y_1[L1-w+b1:L1,:]
test_y_2=test_y_1[L2-w+b2:L2,:]
# validation_y_2=validation_y_1[L3-w+b3:L3,:]



     #### Features ####

train_X=np.concatenate((train_X_1,train_X_2),axis=0)
test_X=np.concatenate((test_X_1,test_X_2),axis=0)
# validation_X=np.concatenate((validation_X_1,validation_X_2),axis=0)


    #### Output ####

train_y=np.concatenate((train_y_1,train_y_2),axis=0)
test_y=np.concatenate((test_y_1,test_y_2),axis=0)
# validation_y=np.concatenate((validation_y_1,validation_y_2),axis=0)


print(train_y.shape)
    #### Reshaping ####
train_X_3_p= train_X.reshape((a1+1,w,train_X.shape[1]))
test_X = test_X.reshape((a2+1,w,test_X.shape[1]))


train_y_3_p= train_y.reshape((a1+1,w,5))
test_y= test_y.reshape((a2+1,w,5))



# train_X_1D=train_X_3
test_X_1D=test_X

train_X_3=train_X_3_p
train_y_3=train_y_3_p
# print(train_X_4.shape,train_y_3.shape)


train_X_1D, X_validation_1D, train_y_5, Y_validation = train_test_split(train_X_3,train_y_3, test_size=0.20, random_state=True)
#train_X_1D, X_validation_1D_ridge, train_y, Y_validation_ridge = train_test_split(train_X_1D_m,train_y_m, test_size=0.10, random_state=True)   [0:2668,:,:]

print(train_X_1D.shape,train_y_5.shape,X_validation_1D.shape,Y_validation.shape)


s=test_X_1D.shape[0]*w

gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()

# from numpy import savetxt
# savetxt('train_data_check.csv', train_dataset_1[:,48:92], delimiter=',')

### IMUs- Chest, Waist, Right Foot, Right shank, Right thigh, Left Foot, Left shank, Left thigh, 2D-body coordinate
### 0:48- IMU, 48:92-2D body coordinate, 92:97-- Target


### Data Processing

batch_size = 64

val_targets = torch.Tensor(Y_validation)
test_features = torch.Tensor(test_X_1D)
test_targets = torch.Tensor(test_y)


## all Modality Features

train_features = torch.Tensor(train_X_1D)
train_targets = torch.Tensor(train_y_5)
val_features = torch.Tensor(X_validation_1D)


train_features_acc_8=torch.cat((train_features[:,:,0:3],train_features[:,:,6:9],train_features[:,:,12:15],train_features[:,:,18:21],train_features[:,:,24:27]\
                             ,train_features[:,:,30:33],train_features[:,:,36:39],train_features[:,:,42:45]),axis=-1)
test_features_acc_8=torch.cat((test_features[:,:,0:3],test_features[:,:,6:9],test_features[:,:,12:15],test_features[:,:,18:21],test_features[:,:,24:27]\
                             ,test_features[:,:,30:33],test_features[:,:,36:39],test_features[:,:,42:45]),axis=-1)
val_features_acc_8=torch.cat((val_features[:,:,0:3],val_features[:,:,6:9],val_features[:,:,12:15],val_features[:,:,18:21],val_features[:,:,24:27]\
                             ,val_features[:,:,30:33],val_features[:,:,36:39],val_features[:,:,42:45]),axis=-1)


train_features_gyr_8=torch.cat((train_features[:,:,3:6],train_features[:,:,9:12],train_features[:,:,15:18],train_features[:,:,21:24],train_features[:,:,27:30]\
                             ,train_features[:,:,33:36],train_features[:,:,39:42],train_features[:,:,45:48]),axis=-1)
test_features_gyr_8=torch.cat((test_features[:,:,3:6],test_features[:,:,9:12],test_features[:,:,15:18],test_features[:,:,21:24],test_features[:,:,27:30]\
                             ,test_features[:,:,33:36],test_features[:,:,39:42],test_features[:,:,45:48]),axis=-1)
val_features_gyr_8=torch.cat((val_features[:,:,3:6],val_features[:,:,9:12],val_features[:,:,15:18],val_features[:,:,21:24],val_features[:,:,27:30]\
                             ,val_features[:,:,33:36],val_features[:,:,39:42],val_features[:,:,45:48]),axis=-1)


train_features_2D_point=train_features[:,:,48:92]
test_features_2D_point=test_features[:,:,48:92]
val_features_2D_point=val_features[:,:,48:92]


train_features_2D_velocity=train_features[:,:,92:136]
test_features_2D_velocity=test_features[:,:,92:136]
val_features_2D_velocity=val_features[:,:,92:136]


train_features_2D_acceleration=train_features[:,:,136:180]
test_features_2D_acceleration=test_features[:,:,136:180]
val_features_2D_acceleration=val_features[:,:,136:180]


train = TensorDataset(train_features, train_features_acc_8,train_features_gyr_8, train_features_2D_point,train_features_2D_velocity,train_features_2D_acceleration, train_targets)
val = TensorDataset(val_features, val_features_acc_8, val_features_gyr_8, val_features_2D_point, val_features_2D_velocity, val_features_2D_acceleration, val_targets)
test = TensorDataset(test_features, test_features_acc_8, test_features_gyr_8, test_features_2D_point,test_features_2D_velocity,test_features_2D_acceleration, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=False)


# Important Functions

def RMSE_prediction(yhat_4,test_y,s):

  s1=yhat_4.shape[0]*yhat_4.shape[1]

  test_o=test_y.reshape((s1,5))
  yhat=yhat_4.reshape((s1,5))




  y_1_no=yhat[:,0]
  y_2_no=yhat[:,1]
  y_3_no=yhat[:,2]
  y_4_no=yhat[:,3]
  y_5_no=yhat[:,4]
  # y_6_no=yhat[:,5]
  # y_7_no=yhat[:,6]
  #y_8_no=yhat[:,7]
  #y_9_no=yhat[:,8]
  #y_10_no=yhat[:,9]


  y_1=y_1_no
  y_2=y_2_no
  y_3=y_3_no
  y_4=y_4_no
  y_5=y_5_no



  y_test_1=test_o[:,0]
  y_test_2=test_o[:,1]
  y_test_3=test_o[:,2]
  y_test_4=test_o[:,3]
  y_test_5=test_o[:,4]
  # y_test_6=test_o[:,5]
  # y_test_7=test_o[:,6]
  #y_test_8=test_o[:,7]
  #y_test_9=test_o[:,8]
  #y_test_10=test_o[:,9]





  #print(y_1.shape,y_test_1.shape)




  Z_1=y_1
  Z_2=y_2
  Z_3=y_3
  Z_4=y_4
  Z_5=y_5
  # Z_6=y_6
  # Z_7=y_7
  #Z_8=y_8
  #Z_9=y_9
  #Z_10=y_10



  ###calculate RMSE

  rmse_1 =((np.sqrt(mean_squared_error(y_test_1,y_1)))/(max(y_test_1)-min(y_test_1)))*100
  rmse_2 =((np.sqrt(mean_squared_error(y_test_2,y_2)))/(max(y_test_2)-min(y_test_2)))*100
  rmse_3 =((np.sqrt(mean_squared_error(y_test_3,y_3)))/(max(y_test_3)-min(y_test_3)))*100
  rmse_4 =((np.sqrt(mean_squared_error(y_test_4,y_4)))/(max(y_test_4)-min(y_test_4)))*100
  rmse_5 =((np.sqrt(mean_squared_error(y_test_5,y_5)))/(max(y_test_5)-min(y_test_5)))*100
  # rmse_6 =((np.sqrt(mean_squared_error(y_test_6,y_6)))/(max(y_test_6)-min(y_test_6)))*100
  # rmse_7 =((np.sqrt(mean_squared_error(y_test_7,y_7)))/(max(y_test_7)-min(y_test_7)))*100
  #rmse_8 =((np.sqrt(mean_squared_error(y_test_8,y_8)))/(max(y_test_8)-min(y_test_8)))*100
  #rmse_9 =((np.sqrt(mean_squared_error(y_test_9,y_9)))/(max(y_test_9)-min(y_test_9)))*100
  #rmse_10 =((np.sqrt(mean_squared_error(y_test_10,y_10)))/(max(y_test_10)-min(y_test_10)))*100


  print(rmse_1)
  print(rmse_2)
  print(rmse_3)
  print(rmse_4)
  print(rmse_5)
  # print(rmse_6)
  # print(rmse_7)
  #print(rmse_8)
  #print(rmse_9)
  #print(rmse_10)


  p_1=np.corrcoef(y_1, y_test_1)[0, 1]
  p_2=np.corrcoef(y_2, y_test_2)[0, 1]
  p_3=np.corrcoef(y_3, y_test_3)[0, 1]
  p_4=np.corrcoef(y_4, y_test_4)[0, 1]
  p_5=np.corrcoef(y_5, y_test_5)[0, 1]
  # p_6=np.corrcoef(y_6, y_test_6)[0, 1]
  # p_7=np.corrcoef(y_7, y_test_7)[0, 1]
  #p_8=np.corrcoef(y_8, y_test_8)[0, 1]
  #p_9=np.corrcoef(y_9, y_test_9)[0, 1]
  #p_10=np.corrcoef(y_10, y_test_10)[0, 1]


  print("\n")
  print(p_1)
  print(p_2)
  print(p_3)
  print(p_4)
  print(p_5)
  # print(p_6)
  # print(p_7)
  #print(p_8)
  #print(p_9)
  #print(p_10)


              ### Correlation ###
  p=np.array([p_1,p_2,p_3,p_4,p_5])




      #### Mean and standard deviation ####

  rmse=np.array([rmse_1,rmse_2,rmse_3,rmse_4,rmse_5])

      #### Mean and standard deviation ####
  m=statistics.mean(rmse)
  SD=statistics.stdev(rmse)
  print('Mean: %.3f' % m,'+/- %.3f' %SD)

  m_c=statistics.mean(p)
  SD_c=statistics.stdev(p)
  print('Mean: %.3f' % m_c,'+/- %.3f' %SD_c)



  return rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5


############################################################################################################################################################################################################################################################################################################################################################################################################################################################################


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target):
        mse = nn.MSELoss()(pred, target)
        rmse = torch.sqrt(mse)
        return rmse

    
############################################################################################################################################################################################################################################################################################################################################################################################################################################################################# Training Function




# Vanilla Knowledge Distillation

class GatingModule(nn.Module):
    def __init__(self, input_size):
        super(GatingModule, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(2*input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        # Apply gating mechanism
        gate_output = self.gate(torch.cat((input1,input2),dim=-1))

        # Scale the inputs based on the gate output
        gated_input1 = input1 * gate_output
        gated_input2 = input2 * (1 - gate_output)

        # Combine the gated inputs
        output = gated_input1 + gated_input2
        return output

class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True, dropout=0.0)
        self.lstm_2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True, dropout=0.0)
        self.flatten=nn.Flatten()
        self.fc = nn.Linear(128, 32)
        self.dropout_1=nn.Dropout(dropout)
        self.dropout_2=nn.Dropout(dropout)


    def forward(self, x):
        out_1, _ = self.lstm_1(x)
        out_1=self.dropout_1(out_1)
        out_2, _ = self.lstm_2(out_1)
        out_2=self.dropout_2(out_2)

        return out_2


adjacency_matrix = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adjacency_matrix):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adjacency_matrix, support) + self.bias
        return output

class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(GraphConvolutionalNetwork, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.dropout=nn.Dropout(p=0.10)
        self.attention = nn.Linear(in_features, 1)

    def forward(self, x, adjacency_matrix):

        attention_weights = self.attention(x)
        attention_weights = F.softmax(attention_weights, dim=1)
        # Apply the attention weights to the input features.
        weighted_features = attention_weights * x


        x = F.relu(self.gc1(weighted_features, adjacency_matrix))
        x=self.dropout(x)
        # x = F.relu(self.gc2(x, adjacency_matrix))
        # x=self.dropout(x)
        return x

## Teacher Model

## Teacher Model

def train_mm_m(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    # criterion =nn.MSELoss()
    criterion =RMSELoss()


    # criterion=PearsonCorrLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    optimizer = torch.optim.Adam(model.parameters())


    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output,x= model(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())

            loss = criterion(output, target.to(device).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target in val_loader:
                output,x= model(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())
                val_loss += criterion(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")


    # # Save the trained model
    # torch.save(model.state_dict(), "model.pth")

    return model


### Weighted Fusion of MHA+Weighted Feature+ Tensor Multiplication Fusion

class MM_mha_wf_tm_fusion(nn.Module):
    def __init__(self, input_acc, input_gyr,input_2D, drop_prob=0.25):
        super(MM_mha_wf_tm_fusion, self).__init__()

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_2d=Encoder(input_2D, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_2d= nn.BatchNorm1d(input_2D, affine=False)

        

        self.fc = nn.Linear(2*3*128+128,5)
        self.dropout=nn.Dropout(p=0.05)
        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)
        self.fc_gc = nn.Linear(44, 128)

        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())
        
        self.fc_kd = nn.Linear(3*128, 3*128)

    def forward(self, x_acc, x_gyr, x_2d):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_2d_1=x_2d.view(x_2d.size(0)*x_2d.size(1),x_2d.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_2d_1=self.BN_2d(x_2d_1)

        # x_2D_3=x_2d_1.view(-1, 11, 4)
        # x_2D_3=self.gcn(x_2D_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        # x_2D_3=x_2D_3.view(x_acc.size(0),w,44)
        # x_2D_3=self.fc_gc(x_2D_3)

        x_acc_2=x_acc_1.view(-1, 50, x_acc_1.size(-1))
        x_gyr_2=x_gyr_1.view(-1, 50, x_gyr_1.size(-1))
        x_2d_2=x_2d_1.view(-1, 50, x_2d_1.size(-1))

        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_2d=self.encoder_2d(x_2d_2)

        # x_2d=self.gate(x_2d,x_2D_3)
        x=torch.cat((x_acc,x_gyr,x_2d),dim=-1)

        x_1=self.fc_kd(x)

        # x_1, attn_output_weights=self.attention(x,x,x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        out_3=x_acc*x_gyr*x_2d

        out_c=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out_c)
        out_f=gating_weights_1*out_c

        out=self.fc(out_f)

        return out, x_1
        
        

lr = 0.001
model = MM_mha_wf_tm_fusion(24,24,44)

#mm_mha_wf_tm_fusion = train_mm_m(train_loader, lr,40,model,path+'_teacher_IMU8_2D.pth')

mm_mha_wf_tm_fusion= MM_mha_wf_tm_fusion(24,24,44)
mm_mha_wf_tm_fusion.load_state_dict(torch.load(path+'_teacher_IMU8_2D.pth'))
mm_mha_wf_tm_fusion.to(device)

mm_mha_wf_tm_fusion.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(test_loader):
        output,x= mm_mha_wf_tm_fusion(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_21=np.hstack([rmse,p])


##############################################################################################################################################################################################################
##############################################################################################################################################################################################################
##############################################################################################################################################################################################################









## Student+Pre-training

import torch
import torch.nn as nn

class AttentionLayerLoss(nn.Module):
    def __init__(self):
        super(AttentionLayerLoss, self).__init__()
        self.hidden_size = 3*128
        self.num_heads = 4
        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)
        self.loss = RMSELoss()

    def forward(self, teacher_features, student_features):
        # Apply multi-head attention mechanism to teacher and student features

        # teacher_output, _ = self.attention(teacher_features, teacher_features, teacher_features)
        # student_output, _ = self.attention(student_features, student_features, student_features)

        # Compute the layer loss using KL divergence
        layer_loss = self.loss(teacher_features, student_features)

        return layer_loss


"""## Knowledge distillation"""

def trainmm_encoder(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()
    # criterion_2 =nn.L1Loss()
    # criterion_2 =nn.MSELoss()
    # criterion_2=contrastive_loss()
    # criterion_2 = nn.KLDivLoss()
    # Instantiate the AttentionLayerLoss module
    attention_loss = AttentionLayerLoss()
    attention_loss = attention_loss.to(device)

    criterion_KD=AttentionLayerLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x_student, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())

            loss=criterion_2(x_student_1,x_teacher_1)

            total_loss= loss

            total_running_loss += total_loss.item()

            total_loss.backward()
            optimizer.step()

            running_loss += loss.item()

        a=total_running_loss/len(train_loader)
        train_loss=running_loss/len(train_loader)

        running_loss=0

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f}")
        torch.save(model.state_dict(), filename)



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")


    return model

class MM_VID_Kinect_mha_wf_tm_fusion_encoder(nn.Module):
    def __init__(self, input_acc, input_gyr, input_2D, drop_prob=0.25):
        super(MM_VID_Kinect_mha_wf_tm_fusion_encoder, self).__init__()

        self.gcn_1=GraphConvolutionalNetwork(4,4)
        self.gcn_2=GraphConvolutionalNetwork(4,4)
        self.gcn_3=GraphConvolutionalNetwork(4,4)

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_2D=Encoder(input_2D, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_2D= nn.BatchNorm1d(input_2D, affine=False)

        self.fc_gc_1 = nn.Linear(44, 128)
        self.fc_gc_2 = nn.Linear(44, 128)
        self.fc_gc_3 = nn.Linear(44, 128)
        
        
        self.dropout=nn.Dropout(p=0.05)
        
        self.gate=GatingModule(128*3)
        
        
        self.fc = nn.Linear(2*3*128+128,5)


        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())

        self.fc_kd = nn.Linear(3*128, 3*128)

    def forward(self, x_acc, x_gyr, x_2D):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_2D_1=x_2D.view(x_2D.size(0)*x_2D.size(1),x_2D.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_2D_1=self.BN_2D(x_2D_1)

        x_acc_3=x_acc_1.view(-1, 11, 4)
        x_gyr_3=x_gyr_1.view(-1, 11, 4)
        x_2D_3=x_2D_1.view(-1, 11, 4)

        ## Graph Convlutional Network

        x_acc_3=self.gcn_1(x_acc_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_gyr_3=self.gcn_2(x_gyr_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_2D_3=self.gcn_3(x_2D_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))

        x_acc_3=x_acc_3.view(x_acc.size(0),w,44)
        x_gyr_3=x_gyr_3.view(x_acc.size(0),w,44)
        x_2D_3=x_2D_3.view(x_acc.size(0),w,44)

        x_acc_3=self.fc_gc_1(x_acc_3)
        x_gyr_3=self.fc_gc_2(x_gyr_3)
        x_2D_3=self.fc_gc_3(x_2D_3)

        x_acc_2=x_acc_1.view(x_acc.size(0),w,44)
        x_gyr_2=x_gyr_1.view(x_acc.size(0),w,44)
        x_2D_2=x_2D_1.view(x_acc.size(0),w,44)

        #### Bi-LSTM Encoder
        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_2D=self.encoder_2D(x_2D_2)

        x=torch.cat((x_acc,x_gyr,x_2D),dim=-1)
        x_gc=torch.cat((x_acc_3,x_gyr_3,x_2D_3),dim=-1)
        x=self.gate(x,x_gc)

        x_1=self.fc_kd(x)

        # x_1, attn_output_weights=self.attention(x,x,x)

        # out_1, attn_output_weights=self.attention(x,x,x)

        # gating_weights = self.gating_net(x)
        # out_2=gating_weights*x

        # out_3=x[:,:,0:128]*x[:,:,128:2*128]*x[:,:,2*128:3*128]

        # out=torch.cat((out_1,out_2,out_3),dim=-1)

        # gating_weights_1 = self.gating_net_1(out)
        # out=gating_weights_1*out

        # out=self.fc(out)

        return x, x_1


lr = 0.001
student = MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)

teacher= MM_mha_wf_tm_fusion(24,24,44)
teacher.load_state_dict(torch.load(path+'_teacher_IMU8_2D.pth'))
teacher.to(device)



student_KD= trainmm_encoder(train_loader, lr,30, student,path+'best_model_student_encoder_KD.pth', teacher)

## Student--Fine Tuning

## Student--Fine Tuning

def train_mm_2D_vel_acc_fine_tuning(train_loader, learn_rate, EPOCHS, model,filename):

    if torch.cuda.is_available():
      model.cuda()
    # Defining loss function and optimizer
    # criterion =nn.MSELoss()
    criterion =RMSELoss()
    # criterion =PearsonCorrCoefLoss()

    # criterion=PearsonCorrLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    optimizer = torch.optim.Adam(model.parameters())


    running_loss=0
    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())

            loss = criterion(output, target.to(device).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target in val_loader:
                output= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
                val_loss += criterion(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model

class MM_VID_Kinect_mha_wf_tm_fusion_KD(nn.Module):
    def __init__(self, model1, input_acc, input_gyr, input_2D, drop_prob=0.25):
        super(MM_VID_Kinect_mha_wf_tm_fusion_KD, self).__init__()

        self.model1 = model1

        self.fc = nn.Linear(2*3*128+128,5)

        self.dropout=nn.Dropout(p=0.05)

        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)

        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())


    def forward(self, x_acc, x_gyr, x_2D):

        x,x_1 = self.model1(x_acc, x_gyr, x_2D)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        out_3=x[:,:,0:128]*x[:,:,128:2*128]*x[:,:,2*128:3*128]

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out_f=gating_weights_1*out

        out=self.fc(out_f)

        return out

student_encoder= MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)
student_encoder.load_state_dict(torch.load(path+'best_model_student_encoder_KD.pth'))
student_encoder.to(device)

# Freeze the weights of model1
for param in student_encoder.parameters():
    param.requires_grad = False


lr = 0.001
model = MM_VID_Kinect_mha_wf_tm_fusion_KD(student_encoder,44,44,44)

mm_vid_kinect_mha_wf_tm_fusion_KD= train_mm_2D_vel_acc_fine_tuning(train_loader, lr,40,model,path+'best_model_student_KD.pth')

mm_vid_kinect_mha_wf_tm_fusion_KD= MM_VID_Kinect_mha_wf_tm_fusion_KD(student_encoder,44,44,44)
mm_vid_kinect_mha_wf_tm_fusion_KD.load_state_dict(torch.load(path+'best_model_student_KD.pth'))
mm_vid_kinect_mha_wf_tm_fusion_KD.to(device)

mm_vid_kinect_mha_wf_tm_fusion_KD.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(test_loader):
        output= mm_vid_kinect_mha_wf_tm_fusion_KD(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_22=np.hstack([rmse,p])


##########################################################################################################################################################################################
##########################################################################################################################################################################################


#### Result Summary
lstm_result_IMU8_2D=np.vstack([ablation_21,ablation_22])



path_1='/home/sanzidpr/Journal_4/Dataset_B_model_results/Results/'

from numpy import savetxt
savetxt(path_1+subject+'_KD_ablation_results.csv', lstm_result_IMU8_2D, delimiter=',')
       









import sys
sys.exit()



























## Knowledge Distillation

def trainmm_KD(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()


    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(train_loader):
            optimizer.zero_grad()
            student_output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())

            loss=criterion_2(student_output,target.to(device))+0.10*criterion_2(student_output,teacher_output)+0.10*criterion_2(x_student_1,x_teacher_1)
            
            loss_1=criterion_2(student_output,target.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target in val_loader:
                output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model


class MM_VID_Kinect_mha_wf_tm_fusion_encoder(nn.Module):
    def __init__(self, input_acc, input_gyr, input_2D, drop_prob=0.25):
        super(MM_VID_Kinect_mha_wf_tm_fusion_encoder, self).__init__()

        self.gcn_1=GraphConvolutionalNetwork(4,4)
        self.gcn_2=GraphConvolutionalNetwork(4,4)
        self.gcn_3=GraphConvolutionalNetwork(4,4)

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_2D=Encoder(input_2D, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_2D= nn.BatchNorm1d(input_2D, affine=False)

        self.fc_gc_1 = nn.Linear(44, 128)
        self.fc_gc_2 = nn.Linear(44, 128)
        self.fc_gc_3 = nn.Linear(44, 128)
        
        
        self.dropout=nn.Dropout(p=0.05)
        
        self.gate=GatingModule(128*3)
        
        
        self.fc = nn.Linear(2*3*128+128,5)


        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())

        self.fc_kd = nn.Linear(3*128, 3*128)

    def forward(self, x_acc, x_gyr, x_2D):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_2D_1=x_2D.view(x_2D.size(0)*x_2D.size(1),x_2D.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_2D_1=self.BN_2D(x_2D_1)

        x_acc_3=x_acc_1.view(-1, 11, 4)
        x_gyr_3=x_gyr_1.view(-1, 11, 4)
        x_2D_3=x_2D_1.view(-1, 11, 4)

        ## Graph Convlutional Network

        x_acc_3=self.gcn_1(x_acc_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_gyr_3=self.gcn_2(x_gyr_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_2D_3=self.gcn_3(x_2D_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))

        x_acc_3=x_acc_3.view(x_acc.size(0),w,44)
        x_gyr_3=x_gyr_3.view(x_acc.size(0),w,44)
        x_2D_3=x_2D_3.view(x_acc.size(0),w,44)

        x_acc_3=self.fc_gc_1(x_acc_3)
        x_gyr_3=self.fc_gc_2(x_gyr_3)
        x_2D_3=self.fc_gc_3(x_2D_3)

        x_acc_2=x_acc_1.view(x_acc.size(0),w,44)
        x_gyr_2=x_gyr_1.view(x_acc.size(0),w,44)
        x_2D_2=x_2D_1.view(x_acc.size(0),w,44)

        #### Bi-LSTM Encoder
        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_2D=self.encoder_2D(x_2D_2)

        x=torch.cat((x_acc,x_gyr,x_2D),dim=-1)
        x_gc=torch.cat((x_acc_3,x_gyr_3,x_2D_3),dim=-1)
        x=self.gate(x,x_gc)

        x_1=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        out_3=x[:,:,0:128]*x[:,:,128:2*128]*x[:,:,2*128:3*128]

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out_f=gating_weights_1*out

        out=self.fc(out_f)

        return out, x_1


lr = 0.001
student = MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)

teacher= MM_mha_wf_tm_fusion(24,24,44)
teacher.load_state_dict(torch.load(path+'_teacher_IMU8_2D.pth'))
teacher.to(device)

student_KD= trainmm_KD(train_loader, lr,40, student,path+'model_student_KD_1.pth', teacher)

mm_vid_kinect_mha_wf_tm_fusion_KD= MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)
mm_vid_kinect_mha_wf_tm_fusion_KD.load_state_dict(torch.load(path+'model_student_KD_1.pth'))
mm_vid_kinect_mha_wf_tm_fusion_KD.to(device)

mm_vid_kinect_mha_wf_tm_fusion_KD.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(test_loader):
        output,student= mm_vid_kinect_mha_wf_tm_fusion_KD(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_1=np.hstack([rmse,p])


##########################################################################################################################################################################################
##########################################################################################################################################################################################


##############################################################################################################################################################################################################
##############################################################################################################################################################################################################
##############################################################################################################################################################################################################


## Knowledge Distillation

def trainmm_KD(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()


    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(train_loader):
            optimizer.zero_grad()
            student_output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())

            loss=criterion_2(student_output,target.to(device))+0.20*criterion_2(student_output,teacher_output)+0.20*criterion_2(x_student_1,x_teacher_1)
            
            loss_1=criterion_2(student_output,target.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target in val_loader:
                output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model


class MM_VID_Kinect_mha_wf_tm_fusion_encoder(nn.Module):
    def __init__(self, input_acc, input_gyr, input_2D, drop_prob=0.25):
        super(MM_VID_Kinect_mha_wf_tm_fusion_encoder, self).__init__()

        self.gcn_1=GraphConvolutionalNetwork(4,4)
        self.gcn_2=GraphConvolutionalNetwork(4,4)
        self.gcn_3=GraphConvolutionalNetwork(4,4)

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_2D=Encoder(input_2D, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_2D= nn.BatchNorm1d(input_2D, affine=False)

        self.fc_gc_1 = nn.Linear(44, 128)
        self.fc_gc_2 = nn.Linear(44, 128)
        self.fc_gc_3 = nn.Linear(44, 128)
        
        
        self.dropout=nn.Dropout(p=0.05)
        
        self.gate=GatingModule(128*3)
        
        
        self.fc = nn.Linear(2*3*128+128,5)


        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())

        self.fc_kd = nn.Linear(3*128, 3*128)

    def forward(self, x_acc, x_gyr, x_2D):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_2D_1=x_2D.view(x_2D.size(0)*x_2D.size(1),x_2D.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_2D_1=self.BN_2D(x_2D_1)

        x_acc_3=x_acc_1.view(-1, 11, 4)
        x_gyr_3=x_gyr_1.view(-1, 11, 4)
        x_2D_3=x_2D_1.view(-1, 11, 4)

        ## Graph Convlutional Network

        x_acc_3=self.gcn_1(x_acc_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_gyr_3=self.gcn_2(x_gyr_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_2D_3=self.gcn_3(x_2D_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))

        x_acc_3=x_acc_3.view(x_acc.size(0),w,44)
        x_gyr_3=x_gyr_3.view(x_acc.size(0),w,44)
        x_2D_3=x_2D_3.view(x_acc.size(0),w,44)

        x_acc_3=self.fc_gc_1(x_acc_3)
        x_gyr_3=self.fc_gc_2(x_gyr_3)
        x_2D_3=self.fc_gc_3(x_2D_3)

        x_acc_2=x_acc_1.view(x_acc.size(0),w,44)
        x_gyr_2=x_gyr_1.view(x_acc.size(0),w,44)
        x_2D_2=x_2D_1.view(x_acc.size(0),w,44)

        #### Bi-LSTM Encoder
        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_2D=self.encoder_2D(x_2D_2)

        x=torch.cat((x_acc,x_gyr,x_2D),dim=-1)
        x_gc=torch.cat((x_acc_3,x_gyr_3,x_2D_3),dim=-1)
        x=self.gate(x,x_gc)

        x_1=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        out_3=x[:,:,0:128]*x[:,:,128:2*128]*x[:,:,2*128:3*128]

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out_f=gating_weights_1*out

        out=self.fc(out_f)

        return out, x_1


lr = 0.001
student = MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)

teacher= MM_mha_wf_tm_fusion(24,24,44)
teacher.load_state_dict(torch.load(path+'_teacher_IMU8_2D.pth'))
teacher.to(device)

student_KD= trainmm_KD(train_loader, lr,40, student,path+'model_student_KD_2.pth', teacher)

mm_vid_kinect_mha_wf_tm_fusion_KD= MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)
mm_vid_kinect_mha_wf_tm_fusion_KD.load_state_dict(torch.load(path+'model_student_KD_2.pth'))
mm_vid_kinect_mha_wf_tm_fusion_KD.to(device)

mm_vid_kinect_mha_wf_tm_fusion_KD.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(test_loader):
        output,student= mm_vid_kinect_mha_wf_tm_fusion_KD(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_2=np.hstack([rmse,p])


##########################################################################################################################################################################################
##########################################################################################################################################################################################


##############################################################################################################################################################################################################
##############################################################################################################################################################################################################
##############################################################################################################################################################################################################


## Knowledge Distillation

def trainmm_KD(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()


    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(train_loader):
            optimizer.zero_grad()
            student_output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())

            loss=criterion_2(student_output,target.to(device))+0.30*criterion_2(student_output,teacher_output)+0.30*criterion_2(x_student_1,x_teacher_1)
            
            loss_1=criterion_2(student_output,target.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target in val_loader:
                output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model


class MM_VID_Kinect_mha_wf_tm_fusion_encoder(nn.Module):
    def __init__(self, input_acc, input_gyr, input_2D, drop_prob=0.25):
        super(MM_VID_Kinect_mha_wf_tm_fusion_encoder, self).__init__()

        self.gcn_1=GraphConvolutionalNetwork(4,4)
        self.gcn_2=GraphConvolutionalNetwork(4,4)
        self.gcn_3=GraphConvolutionalNetwork(4,4)

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_2D=Encoder(input_2D, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_2D= nn.BatchNorm1d(input_2D, affine=False)

        self.fc_gc_1 = nn.Linear(44, 128)
        self.fc_gc_2 = nn.Linear(44, 128)
        self.fc_gc_3 = nn.Linear(44, 128)
        
        
        self.dropout=nn.Dropout(p=0.05)
        
        self.gate=GatingModule(128*3)
        
        
        self.fc = nn.Linear(2*3*128+128,5)


        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())

        self.fc_kd = nn.Linear(3*128, 3*128)

    def forward(self, x_acc, x_gyr, x_2D):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_2D_1=x_2D.view(x_2D.size(0)*x_2D.size(1),x_2D.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_2D_1=self.BN_2D(x_2D_1)

        x_acc_3=x_acc_1.view(-1, 11, 4)
        x_gyr_3=x_gyr_1.view(-1, 11, 4)
        x_2D_3=x_2D_1.view(-1, 11, 4)

        ## Graph Convlutional Network

        x_acc_3=self.gcn_1(x_acc_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_gyr_3=self.gcn_2(x_gyr_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_2D_3=self.gcn_3(x_2D_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))

        x_acc_3=x_acc_3.view(x_acc.size(0),w,44)
        x_gyr_3=x_gyr_3.view(x_acc.size(0),w,44)
        x_2D_3=x_2D_3.view(x_acc.size(0),w,44)

        x_acc_3=self.fc_gc_1(x_acc_3)
        x_gyr_3=self.fc_gc_2(x_gyr_3)
        x_2D_3=self.fc_gc_3(x_2D_3)

        x_acc_2=x_acc_1.view(x_acc.size(0),w,44)
        x_gyr_2=x_gyr_1.view(x_acc.size(0),w,44)
        x_2D_2=x_2D_1.view(x_acc.size(0),w,44)

        #### Bi-LSTM Encoder
        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_2D=self.encoder_2D(x_2D_2)

        x=torch.cat((x_acc,x_gyr,x_2D),dim=-1)
        x_gc=torch.cat((x_acc_3,x_gyr_3,x_2D_3),dim=-1)
        x=self.gate(x,x_gc)

        x_1=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        out_3=x[:,:,0:128]*x[:,:,128:2*128]*x[:,:,2*128:3*128]

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out_f=gating_weights_1*out

        out=self.fc(out_f)

        return out, x_1


lr = 0.001
student = MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)

teacher= MM_mha_wf_tm_fusion(24,24,44)
teacher.load_state_dict(torch.load(path+'_teacher_IMU8_2D.pth'))
teacher.to(device)

student_KD= trainmm_KD(train_loader, lr,40, student,path+'model_student_KD_3.pth', teacher)

mm_vid_kinect_mha_wf_tm_fusion_KD= MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)
mm_vid_kinect_mha_wf_tm_fusion_KD.load_state_dict(torch.load(path+'model_student_KD_3.pth'))
mm_vid_kinect_mha_wf_tm_fusion_KD.to(device)

mm_vid_kinect_mha_wf_tm_fusion_KD.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(test_loader):
        output,student= mm_vid_kinect_mha_wf_tm_fusion_KD(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_3=np.hstack([rmse,p])


##########################################################################################################################################################################################
##########################################################################################################################################################################################


##############################################################################################################################################################################################################
##############################################################################################################################################################################################################
##############################################################################################################################################################################################################


## Knowledge Distillation

def trainmm_KD(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()


    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(train_loader):
            optimizer.zero_grad()
            student_output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())

            loss=criterion_2(student_output,target.to(device))+0.40*criterion_2(student_output,teacher_output)+0.40*criterion_2(x_student_1,x_teacher_1)
            
            loss_1=criterion_2(student_output,target.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target in val_loader:
                output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model


class MM_VID_Kinect_mha_wf_tm_fusion_encoder(nn.Module):
    def __init__(self, input_acc, input_gyr, input_2D, drop_prob=0.25):
        super(MM_VID_Kinect_mha_wf_tm_fusion_encoder, self).__init__()

        self.gcn_1=GraphConvolutionalNetwork(4,4)
        self.gcn_2=GraphConvolutionalNetwork(4,4)
        self.gcn_3=GraphConvolutionalNetwork(4,4)

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_2D=Encoder(input_2D, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_2D= nn.BatchNorm1d(input_2D, affine=False)

        self.fc_gc_1 = nn.Linear(44, 128)
        self.fc_gc_2 = nn.Linear(44, 128)
        self.fc_gc_3 = nn.Linear(44, 128)
        
        
        self.dropout=nn.Dropout(p=0.05)
        
        self.gate=GatingModule(128*3)
        
        
        self.fc = nn.Linear(2*3*128+128,5)


        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())

        self.fc_kd = nn.Linear(3*128, 3*128)

    def forward(self, x_acc, x_gyr, x_2D):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_2D_1=x_2D.view(x_2D.size(0)*x_2D.size(1),x_2D.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_2D_1=self.BN_2D(x_2D_1)

        x_acc_3=x_acc_1.view(-1, 11, 4)
        x_gyr_3=x_gyr_1.view(-1, 11, 4)
        x_2D_3=x_2D_1.view(-1, 11, 4)

        ## Graph Convlutional Network

        x_acc_3=self.gcn_1(x_acc_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_gyr_3=self.gcn_2(x_gyr_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_2D_3=self.gcn_3(x_2D_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))

        x_acc_3=x_acc_3.view(x_acc.size(0),w,44)
        x_gyr_3=x_gyr_3.view(x_acc.size(0),w,44)
        x_2D_3=x_2D_3.view(x_acc.size(0),w,44)

        x_acc_3=self.fc_gc_1(x_acc_3)
        x_gyr_3=self.fc_gc_2(x_gyr_3)
        x_2D_3=self.fc_gc_3(x_2D_3)

        x_acc_2=x_acc_1.view(x_acc.size(0),w,44)
        x_gyr_2=x_gyr_1.view(x_acc.size(0),w,44)
        x_2D_2=x_2D_1.view(x_acc.size(0),w,44)

        #### Bi-LSTM Encoder
        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_2D=self.encoder_2D(x_2D_2)

        x=torch.cat((x_acc,x_gyr,x_2D),dim=-1)
        x_gc=torch.cat((x_acc_3,x_gyr_3,x_2D_3),dim=-1)
        x=self.gate(x,x_gc)

        x_1=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        out_3=x[:,:,0:128]*x[:,:,128:2*128]*x[:,:,2*128:3*128]

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out_f=gating_weights_1*out

        out=self.fc(out_f)

        return out, x_1


lr = 0.001
student = MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)

teacher= MM_mha_wf_tm_fusion(24,24,44)
teacher.load_state_dict(torch.load(path+'_teacher_IMU8_2D.pth'))
teacher.to(device)

student_KD= trainmm_KD(train_loader, lr,40, student,path+'model_student_KD_4.pth', teacher)

mm_vid_kinect_mha_wf_tm_fusion_KD= MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)
mm_vid_kinect_mha_wf_tm_fusion_KD.load_state_dict(torch.load(path+'model_student_KD_4.pth'))
mm_vid_kinect_mha_wf_tm_fusion_KD.to(device)

mm_vid_kinect_mha_wf_tm_fusion_KD.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(test_loader):
        output,student= mm_vid_kinect_mha_wf_tm_fusion_KD(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_4=np.hstack([rmse,p])


##########################################################################################################################################################################################
##########################################################################################################################################################################################


##############################################################################################################################################################################################################
##############################################################################################################################################################################################################
##############################################################################################################################################################################################################


## Knowledge Distillation

def trainmm_KD(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()


    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(train_loader):
            optimizer.zero_grad()
            student_output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())

            loss=criterion_2(student_output,target.to(device))+0.50*criterion_2(student_output,teacher_output)+0.50*criterion_2(x_student_1,x_teacher_1)
            
            loss_1=criterion_2(student_output,target.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target in val_loader:
                output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model


class MM_VID_Kinect_mha_wf_tm_fusion_encoder(nn.Module):
    def __init__(self, input_acc, input_gyr, input_2D, drop_prob=0.25):
        super(MM_VID_Kinect_mha_wf_tm_fusion_encoder, self).__init__()

        self.gcn_1=GraphConvolutionalNetwork(4,4)
        self.gcn_2=GraphConvolutionalNetwork(4,4)
        self.gcn_3=GraphConvolutionalNetwork(4,4)

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_2D=Encoder(input_2D, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_2D= nn.BatchNorm1d(input_2D, affine=False)

        self.fc_gc_1 = nn.Linear(44, 128)
        self.fc_gc_2 = nn.Linear(44, 128)
        self.fc_gc_3 = nn.Linear(44, 128)
        
        
        self.dropout=nn.Dropout(p=0.05)
        
        self.gate=GatingModule(128*3)
        
        
        self.fc = nn.Linear(2*3*128+128,5)


        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())

        self.fc_kd = nn.Linear(3*128, 3*128)

    def forward(self, x_acc, x_gyr, x_2D):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_2D_1=x_2D.view(x_2D.size(0)*x_2D.size(1),x_2D.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_2D_1=self.BN_2D(x_2D_1)

        x_acc_3=x_acc_1.view(-1, 11, 4)
        x_gyr_3=x_gyr_1.view(-1, 11, 4)
        x_2D_3=x_2D_1.view(-1, 11, 4)

        ## Graph Convlutional Network

        x_acc_3=self.gcn_1(x_acc_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_gyr_3=self.gcn_2(x_gyr_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_2D_3=self.gcn_3(x_2D_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))

        x_acc_3=x_acc_3.view(x_acc.size(0),w,44)
        x_gyr_3=x_gyr_3.view(x_acc.size(0),w,44)
        x_2D_3=x_2D_3.view(x_acc.size(0),w,44)

        x_acc_3=self.fc_gc_1(x_acc_3)
        x_gyr_3=self.fc_gc_2(x_gyr_3)
        x_2D_3=self.fc_gc_3(x_2D_3)

        x_acc_2=x_acc_1.view(x_acc.size(0),w,44)
        x_gyr_2=x_gyr_1.view(x_acc.size(0),w,44)
        x_2D_2=x_2D_1.view(x_acc.size(0),w,44)

        #### Bi-LSTM Encoder
        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_2D=self.encoder_2D(x_2D_2)

        x=torch.cat((x_acc,x_gyr,x_2D),dim=-1)
        x_gc=torch.cat((x_acc_3,x_gyr_3,x_2D_3),dim=-1)
        x=self.gate(x,x_gc)

        x_1=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        out_3=x[:,:,0:128]*x[:,:,128:2*128]*x[:,:,2*128:3*128]

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out_f=gating_weights_1*out

        out=self.fc(out_f)

        return out, x_1


lr = 0.001
student = MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)

teacher= MM_mha_wf_tm_fusion(24,24,44)
teacher.load_state_dict(torch.load(path+'_teacher_IMU8_2D.pth'))
teacher.to(device)

student_KD= trainmm_KD(train_loader, lr,40, student,path+'model_student_KD_5.pth', teacher)

mm_vid_kinect_mha_wf_tm_fusion_KD= MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)
mm_vid_kinect_mha_wf_tm_fusion_KD.load_state_dict(torch.load(path+'model_student_KD_5.pth'))
mm_vid_kinect_mha_wf_tm_fusion_KD.to(device)

mm_vid_kinect_mha_wf_tm_fusion_KD.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(test_loader):
        output,student= mm_vid_kinect_mha_wf_tm_fusion_KD(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_5=np.hstack([rmse,p])


##########################################################################################################################################################################################
##########################################################################################################################################################################################



##############################################################################################################################################################################################################
##############################################################################################################################################################################################################
##############################################################################################################################################################################################################


## Knowledge Distillation

def trainmm_KD(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()


    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(train_loader):
            optimizer.zero_grad()
            student_output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())

            loss=criterion_2(student_output,target.to(device))+0.60*criterion_2(student_output,teacher_output)+0.60*criterion_2(x_student_1,x_teacher_1)
            
            loss_1=criterion_2(student_output,target.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target in val_loader:
                output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model


class MM_VID_Kinect_mha_wf_tm_fusion_encoder(nn.Module):
    def __init__(self, input_acc, input_gyr, input_2D, drop_prob=0.25):
        super(MM_VID_Kinect_mha_wf_tm_fusion_encoder, self).__init__()

        self.gcn_1=GraphConvolutionalNetwork(4,4)
        self.gcn_2=GraphConvolutionalNetwork(4,4)
        self.gcn_3=GraphConvolutionalNetwork(4,4)

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_2D=Encoder(input_2D, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_2D= nn.BatchNorm1d(input_2D, affine=False)

        self.fc_gc_1 = nn.Linear(44, 128)
        self.fc_gc_2 = nn.Linear(44, 128)
        self.fc_gc_3 = nn.Linear(44, 128)
        
        
        self.dropout=nn.Dropout(p=0.05)
        
        self.gate=GatingModule(128*3)
        
        
        self.fc = nn.Linear(2*3*128+128,5)


        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())

        self.fc_kd = nn.Linear(3*128, 3*128)

    def forward(self, x_acc, x_gyr, x_2D):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_2D_1=x_2D.view(x_2D.size(0)*x_2D.size(1),x_2D.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_2D_1=self.BN_2D(x_2D_1)

        x_acc_3=x_acc_1.view(-1, 11, 4)
        x_gyr_3=x_gyr_1.view(-1, 11, 4)
        x_2D_3=x_2D_1.view(-1, 11, 4)

        ## Graph Convlutional Network

        x_acc_3=self.gcn_1(x_acc_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_gyr_3=self.gcn_2(x_gyr_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_2D_3=self.gcn_3(x_2D_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))

        x_acc_3=x_acc_3.view(x_acc.size(0),w,44)
        x_gyr_3=x_gyr_3.view(x_acc.size(0),w,44)
        x_2D_3=x_2D_3.view(x_acc.size(0),w,44)

        x_acc_3=self.fc_gc_1(x_acc_3)
        x_gyr_3=self.fc_gc_2(x_gyr_3)
        x_2D_3=self.fc_gc_3(x_2D_3)

        x_acc_2=x_acc_1.view(x_acc.size(0),w,44)
        x_gyr_2=x_gyr_1.view(x_acc.size(0),w,44)
        x_2D_2=x_2D_1.view(x_acc.size(0),w,44)

        #### Bi-LSTM Encoder
        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_2D=self.encoder_2D(x_2D_2)

        x=torch.cat((x_acc,x_gyr,x_2D),dim=-1)
        x_gc=torch.cat((x_acc_3,x_gyr_3,x_2D_3),dim=-1)
        x=self.gate(x,x_gc)

        x_1=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        out_3=x[:,:,0:128]*x[:,:,128:2*128]*x[:,:,2*128:3*128]

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out_f=gating_weights_1*out

        out=self.fc(out_f)

        return out, x_1


lr = 0.001
student = MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)

teacher= MM_mha_wf_tm_fusion(24,24,44)
teacher.load_state_dict(torch.load(path+'_teacher_IMU8_2D.pth'))
teacher.to(device)

student_KD= trainmm_KD(train_loader, lr,40, student,path+'model_student_KD_6.pth', teacher)

mm_vid_kinect_mha_wf_tm_fusion_KD= MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)
mm_vid_kinect_mha_wf_tm_fusion_KD.load_state_dict(torch.load(path+'model_student_KD_6.pth'))
mm_vid_kinect_mha_wf_tm_fusion_KD.to(device)

mm_vid_kinect_mha_wf_tm_fusion_KD.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(test_loader):
        output,student= mm_vid_kinect_mha_wf_tm_fusion_KD(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_6=np.hstack([rmse,p])


##########################################################################################################################################################################################
##########################################################################################################################################################################################



##############################################################################################################################################################################################################
##############################################################################################################################################################################################################
##############################################################################################################################################################################################################


## Knowledge Distillation

def trainmm_KD(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()


    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(train_loader):
            optimizer.zero_grad()
            student_output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())

            loss=criterion_2(student_output,target.to(device))+0.70*criterion_2(student_output,teacher_output)+0.70*criterion_2(x_student_1,x_teacher_1)
            
            loss_1=criterion_2(student_output,target.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target in val_loader:
                output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model


class MM_VID_Kinect_mha_wf_tm_fusion_encoder(nn.Module):
    def __init__(self, input_acc, input_gyr, input_2D, drop_prob=0.25):
        super(MM_VID_Kinect_mha_wf_tm_fusion_encoder, self).__init__()

        self.gcn_1=GraphConvolutionalNetwork(4,4)
        self.gcn_2=GraphConvolutionalNetwork(4,4)
        self.gcn_3=GraphConvolutionalNetwork(4,4)

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_2D=Encoder(input_2D, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_2D= nn.BatchNorm1d(input_2D, affine=False)

        self.fc_gc_1 = nn.Linear(44, 128)
        self.fc_gc_2 = nn.Linear(44, 128)
        self.fc_gc_3 = nn.Linear(44, 128)
        
        
        self.dropout=nn.Dropout(p=0.05)
        
        self.gate=GatingModule(128*3)
        
        
        self.fc = nn.Linear(2*3*128+128,5)


        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())

        self.fc_kd = nn.Linear(3*128, 3*128)

    def forward(self, x_acc, x_gyr, x_2D):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_2D_1=x_2D.view(x_2D.size(0)*x_2D.size(1),x_2D.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_2D_1=self.BN_2D(x_2D_1)

        x_acc_3=x_acc_1.view(-1, 11, 4)
        x_gyr_3=x_gyr_1.view(-1, 11, 4)
        x_2D_3=x_2D_1.view(-1, 11, 4)

        ## Graph Convlutional Network

        x_acc_3=self.gcn_1(x_acc_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_gyr_3=self.gcn_2(x_gyr_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_2D_3=self.gcn_3(x_2D_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))

        x_acc_3=x_acc_3.view(x_acc.size(0),w,44)
        x_gyr_3=x_gyr_3.view(x_acc.size(0),w,44)
        x_2D_3=x_2D_3.view(x_acc.size(0),w,44)

        x_acc_3=self.fc_gc_1(x_acc_3)
        x_gyr_3=self.fc_gc_2(x_gyr_3)
        x_2D_3=self.fc_gc_3(x_2D_3)

        x_acc_2=x_acc_1.view(x_acc.size(0),w,44)
        x_gyr_2=x_gyr_1.view(x_acc.size(0),w,44)
        x_2D_2=x_2D_1.view(x_acc.size(0),w,44)

        #### Bi-LSTM Encoder
        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_2D=self.encoder_2D(x_2D_2)

        x=torch.cat((x_acc,x_gyr,x_2D),dim=-1)
        x_gc=torch.cat((x_acc_3,x_gyr_3,x_2D_3),dim=-1)
        x=self.gate(x,x_gc)

        x_1=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        out_3=x[:,:,0:128]*x[:,:,128:2*128]*x[:,:,2*128:3*128]

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out_f=gating_weights_1*out

        out=self.fc(out_f)

        return out, x_1


lr = 0.001
student = MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)

teacher= MM_mha_wf_tm_fusion(24,24,44)
teacher.load_state_dict(torch.load(path+'_teacher_IMU8_2D.pth'))
teacher.to(device)

student_KD= trainmm_KD(train_loader, lr,40, student,path+'model_student_KD_7.pth', teacher)

mm_vid_kinect_mha_wf_tm_fusion_KD= MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)
mm_vid_kinect_mha_wf_tm_fusion_KD.load_state_dict(torch.load(path+'model_student_KD_7.pth'))
mm_vid_kinect_mha_wf_tm_fusion_KD.to(device)

mm_vid_kinect_mha_wf_tm_fusion_KD.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(test_loader):
        output,student= mm_vid_kinect_mha_wf_tm_fusion_KD(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_7=np.hstack([rmse,p])


##########################################################################################################################################################################################
##########################################################################################################################################################################################


##############################################################################################################################################################################################################
##############################################################################################################################################################################################################
##############################################################################################################################################################################################################


## Knowledge Distillation

def trainmm_KD(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()


    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(train_loader):
            optimizer.zero_grad()
            student_output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())

            loss=criterion_2(student_output,target.to(device))+0.80*criterion_2(student_output,teacher_output)+0.80*criterion_2(x_student_1,x_teacher_1)
            
            loss_1=criterion_2(student_output,target.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target in val_loader:
                output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model


class MM_VID_Kinect_mha_wf_tm_fusion_encoder(nn.Module):
    def __init__(self, input_acc, input_gyr, input_2D, drop_prob=0.25):
        super(MM_VID_Kinect_mha_wf_tm_fusion_encoder, self).__init__()

        self.gcn_1=GraphConvolutionalNetwork(4,4)
        self.gcn_2=GraphConvolutionalNetwork(4,4)
        self.gcn_3=GraphConvolutionalNetwork(4,4)

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_2D=Encoder(input_2D, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_2D= nn.BatchNorm1d(input_2D, affine=False)

        self.fc_gc_1 = nn.Linear(44, 128)
        self.fc_gc_2 = nn.Linear(44, 128)
        self.fc_gc_3 = nn.Linear(44, 128)
        
        
        self.dropout=nn.Dropout(p=0.05)
        
        self.gate=GatingModule(128*3)
        
        
        self.fc = nn.Linear(2*3*128+128,5)


        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())

        self.fc_kd = nn.Linear(3*128, 3*128)

    def forward(self, x_acc, x_gyr, x_2D):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_2D_1=x_2D.view(x_2D.size(0)*x_2D.size(1),x_2D.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_2D_1=self.BN_2D(x_2D_1)

        x_acc_3=x_acc_1.view(-1, 11, 4)
        x_gyr_3=x_gyr_1.view(-1, 11, 4)
        x_2D_3=x_2D_1.view(-1, 11, 4)

        ## Graph Convlutional Network

        x_acc_3=self.gcn_1(x_acc_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_gyr_3=self.gcn_2(x_gyr_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_2D_3=self.gcn_3(x_2D_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))

        x_acc_3=x_acc_3.view(x_acc.size(0),w,44)
        x_gyr_3=x_gyr_3.view(x_acc.size(0),w,44)
        x_2D_3=x_2D_3.view(x_acc.size(0),w,44)

        x_acc_3=self.fc_gc_1(x_acc_3)
        x_gyr_3=self.fc_gc_2(x_gyr_3)
        x_2D_3=self.fc_gc_3(x_2D_3)

        x_acc_2=x_acc_1.view(x_acc.size(0),w,44)
        x_gyr_2=x_gyr_1.view(x_acc.size(0),w,44)
        x_2D_2=x_2D_1.view(x_acc.size(0),w,44)

        #### Bi-LSTM Encoder
        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_2D=self.encoder_2D(x_2D_2)

        x=torch.cat((x_acc,x_gyr,x_2D),dim=-1)
        x_gc=torch.cat((x_acc_3,x_gyr_3,x_2D_3),dim=-1)
        x=self.gate(x,x_gc)

        x_1=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        out_3=x[:,:,0:128]*x[:,:,128:2*128]*x[:,:,2*128:3*128]

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out_f=gating_weights_1*out

        out=self.fc(out_f)

        return out, x_1


lr = 0.001
student = MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)

teacher= MM_mha_wf_tm_fusion(24,24,44)
teacher.load_state_dict(torch.load(path+'_teacher_IMU8_2D.pth'))
teacher.to(device)

student_KD= trainmm_KD(train_loader, lr,40, student,path+'model_student_KD_8.pth', teacher)

mm_vid_kinect_mha_wf_tm_fusion_KD= MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)
mm_vid_kinect_mha_wf_tm_fusion_KD.load_state_dict(torch.load(path+'model_student_KD_8.pth'))
mm_vid_kinect_mha_wf_tm_fusion_KD.to(device)

mm_vid_kinect_mha_wf_tm_fusion_KD.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(test_loader):
        output,student= mm_vid_kinect_mha_wf_tm_fusion_KD(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_8=np.hstack([rmse,p])


##########################################################################################################################################################################################
##########################################################################################################################################################################################



##############################################################################################################################################################################################################
##############################################################################################################################################################################################################
##############################################################################################################################################################################################################


## Knowledge Distillation

def trainmm_KD(train_loader, learn_rate, EPOCHS, model, filename, teacher):

    if torch.cuda.is_available():
      model.cuda()

    # Defining loss function and optimizer
    criterion_2 =RMSELoss()


    optimizer = torch.optim.Adam(model.parameters())

    total_running_loss=0
    running_loss=0

    # Train the model
    start_time = time.time()

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10


    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(train_loader):
            optimizer.zero_grad()
            student_output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())

            with torch.no_grad():
                teacher_output,x_teacher_1= teacher(data_acc.to(device).float(),data_gyr.to(device).float(),data_2D.to(device).float())

            loss=criterion_2(student_output,target.to(device))+0.90*criterion_2(student_output,teacher_output)+0.90*criterion_2(x_student_1,x_teacher_1)
            
            loss_1=criterion_2(student_output,target.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

        train_loss=running_loss/len(train_loader)

       # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target in val_loader:
                output, x_student_1= model(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
                val_loss += criterion_2(output, target.to(device).float())

        val_loss /= len(val_loader)

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time

        torch.set_printoptions(precision=4)

        print(f"Epoch: {epoch+1}, time: {epoch_training_time:.4f}, Training Loss: {train_loss:.4f},  Validation loss: {val_loss:.4f}")

        running_loss=0

        epoch_end_time = time.time()

                # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), filename)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if the validation loss hasn't improved for `patience` epochs
        if patience_counter >= patience:
            print(f"Stopping early after {epoch+1} epochs")
            break



    end_time = time.time()

    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")



    return model


class MM_VID_Kinect_mha_wf_tm_fusion_encoder(nn.Module):
    def __init__(self, input_acc, input_gyr, input_2D, drop_prob=0.25):
        super(MM_VID_Kinect_mha_wf_tm_fusion_encoder, self).__init__()

        self.gcn_1=GraphConvolutionalNetwork(4,4)
        self.gcn_2=GraphConvolutionalNetwork(4,4)
        self.gcn_3=GraphConvolutionalNetwork(4,4)

        self.encoder_acc=Encoder(input_acc, drop_prob)
        self.encoder_gyr=Encoder(input_gyr, drop_prob)
        self.encoder_2D=Encoder(input_2D, drop_prob)

        self.BN_acc= nn.BatchNorm1d(input_acc, affine=False)
        self.BN_gyr= nn.BatchNorm1d(input_gyr, affine=False)
        self.BN_2D= nn.BatchNorm1d(input_2D, affine=False)

        self.fc_gc_1 = nn.Linear(44, 128)
        self.fc_gc_2 = nn.Linear(44, 128)
        self.fc_gc_3 = nn.Linear(44, 128)
        
        
        self.dropout=nn.Dropout(p=0.05)
        
        self.gate=GatingModule(128*3)
        
        
        self.fc = nn.Linear(2*3*128+128,5)


        self.attention=nn.MultiheadAttention(3*128,4,batch_first=True)
        self.gating_net = nn.Sequential(nn.Linear(128*3, 3*128), nn.Sigmoid())
        self.gating_net_1 = nn.Sequential(nn.Linear(2*3*128+128, 2*3*128+128), nn.Sigmoid())

        self.fc_kd = nn.Linear(3*128, 3*128)

    def forward(self, x_acc, x_gyr, x_2D):

        x_acc_1=x_acc.view(x_acc.size(0)*x_acc.size(1),x_acc.size(-1))
        x_gyr_1=x_gyr.view(x_gyr.size(0)*x_gyr.size(1),x_gyr.size(-1))
        x_2D_1=x_2D.view(x_2D.size(0)*x_2D.size(1),x_2D.size(-1))

        x_acc_1=self.BN_acc(x_acc_1)
        x_gyr_1=self.BN_gyr(x_gyr_1)
        x_2D_1=self.BN_2D(x_2D_1)

        x_acc_3=x_acc_1.view(-1, 11, 4)
        x_gyr_3=x_gyr_1.view(-1, 11, 4)
        x_2D_3=x_2D_1.view(-1, 11, 4)

        ## Graph Convlutional Network

        x_acc_3=self.gcn_1(x_acc_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_gyr_3=self.gcn_2(x_gyr_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))
        x_2D_3=self.gcn_3(x_2D_3, torch.from_numpy(adjacency_matrix.astype(np.float32)).to(device))

        x_acc_3=x_acc_3.view(x_acc.size(0),w,44)
        x_gyr_3=x_gyr_3.view(x_acc.size(0),w,44)
        x_2D_3=x_2D_3.view(x_acc.size(0),w,44)

        x_acc_3=self.fc_gc_1(x_acc_3)
        x_gyr_3=self.fc_gc_2(x_gyr_3)
        x_2D_3=self.fc_gc_3(x_2D_3)

        x_acc_2=x_acc_1.view(x_acc.size(0),w,44)
        x_gyr_2=x_gyr_1.view(x_acc.size(0),w,44)
        x_2D_2=x_2D_1.view(x_acc.size(0),w,44)

        #### Bi-LSTM Encoder
        x_acc=self.encoder_acc(x_acc_2)
        x_gyr=self.encoder_gyr(x_gyr_2)
        x_2D=self.encoder_2D(x_2D_2)

        x=torch.cat((x_acc,x_gyr,x_2D),dim=-1)
        x_gc=torch.cat((x_acc_3,x_gyr_3,x_2D_3),dim=-1)
        x=self.gate(x,x_gc)

        x_1=self.fc_kd(x)

        out_1, attn_output_weights=self.attention(x,x,x)

        gating_weights = self.gating_net(x)
        out_2=gating_weights*x

        out_3=x[:,:,0:128]*x[:,:,128:2*128]*x[:,:,2*128:3*128]

        out=torch.cat((out_1,out_2,out_3),dim=-1)

        gating_weights_1 = self.gating_net_1(out)
        out_f=gating_weights_1*out

        out=self.fc(out_f)

        return out, x_1


lr = 0.001
student = MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)

teacher= MM_mha_wf_tm_fusion(24,24,44)
teacher.load_state_dict(torch.load(path+'_teacher_IMU8_2D.pth'))
teacher.to(device)

student_KD= trainmm_KD(train_loader, lr,40, student,path+'model_student_KD_9.pth', teacher)

mm_vid_kinect_mha_wf_tm_fusion_KD= MM_VID_Kinect_mha_wf_tm_fusion_encoder(44,44,44)
mm_vid_kinect_mha_wf_tm_fusion_KD.load_state_dict(torch.load(path+'model_student_KD_9.pth'))
mm_vid_kinect_mha_wf_tm_fusion_KD.to(device)

mm_vid_kinect_mha_wf_tm_fusion_KD.eval()

# iterate through batches of test data
with torch.no_grad():
    for i, (data, data_acc, data_gyr, data_2D, data_velocity, data_acceleration, target) in enumerate(test_loader):
        output,student= mm_vid_kinect_mha_wf_tm_fusion_KD(data_velocity.to(device).float(),data_acceleration.to(device).float(),data_2D.to(device).float())
        if i==0:
          yhat_5=output
          test_target=target

        yhat_5=torch.cat((yhat_5,output),dim=0)
        test_target=torch.cat((test_target,target),dim=0)

        # clear memory
        del data, target,output
        torch.cuda.empty_cache()


yhat_4 = yhat_5.detach().cpu().numpy()
test_target = test_target.detach().cpu().numpy()
print(yhat_4.shape)

rmse, p, Z_1,Z_2,Z_3,Z_4,Z_5=RMSE_prediction(yhat_4,test_target,s)

ablation_9=np.hstack([rmse,p])


##########################################################################################################################################################################################
##########################################################################################################################################################################################




#### Result Summary
lstm_result_IMU8_2D=np.vstack([ablation_1,ablation_2,ablation_3,ablation_4,ablation_5,ablation_6,ablation_7,ablation_8,ablation_9])



path_1='/home/sanzidpr/Journal_4/Dataset_B_model_results/Results/'

from numpy import savetxt
savetxt(path_1+subject+'_Augmentation_KD_ablation_results.csv', lstm_result_IMU8_2D, delimiter=',')
       


