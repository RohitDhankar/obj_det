
# from __future__ import print_function
# import torch

# x = [12,23,34,45,56,67,78]
# print(x)
# print(torch.is_tensor(x))
# print(torch.is_storage(x))
# #
# y = torch.randn(1,2,3,4,5)
# print(torch.is_tensor(y))
# print(torch.is_storage(y))
# print(torch.numel(y))
# print(y.shape) #torch.Size([1, 2, 3, 4, 5])
# #
# t_z_4x4 = torch.zeros(4,4)
# print(torch.numel(t_z_4x4)) # 16
# print(t_z_4x4.shape)



# ## .eye --- same as Identity Matrix --- np.identity()

# t_i_5 = torch.eye(5)
# print(t_i_5.shape) #torch.Size([5, 5])
# print(t_i_5)
# print(torch.numel(t_i_5)) # 25
# t_i_5.reshape(5,5) ## same as above --->> torch.Size([5, 5])
# print(t_i_5)
# #
# t_i_8 = torch.eye(8)
# print(t_i_8.shape) #torch.Size([8, 8])
# print(t_i_8)
# print(torch.numel(t_i_8)) # 64
# t_id_2x32 = t_i_8.reshape(2,32) ## same as above --->> torch.Size([5, 5])
# print(t_id_2x32)
# print(torch.numel(t_id_2x32)) # 64
# print(t_id_2x32.shape) #torch.Size([2, 32])
# #



# CSCI 5561: Assignment #4
# Convolutional Neural Network
# 1 Submission
# • Assignment due: Nov 22 (11:55pm)
# • Individual assignment
# • Up to 2 page summary write-up with resulting visualization (more than 2 page
# assignment will be automatically returned.).
# • Submission through Canvas.
# • Following skeletal functions are already included in the cnn.py file (https://
# www-users.cs.umn.edu/~hspark/csci5561_F2019/HW4.zip)
# – main_slp_linear
# – main_slp
# – main_mlp
# – main_cnn
# • List of function to submit:
# – get_mini_batch
# – fc
# – fc_backward
# – loss_euclidean
# – train_slp_linear
# – loss_cross_entropy_softmax
