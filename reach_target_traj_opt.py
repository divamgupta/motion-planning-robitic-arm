
import numpy as np 
from scipy.optimize import leastsq

from arm import render_arm , forward_kinematics_arm_torch
import scipy
import time

from reach_target_optimization import  get_grid_pts , get_obstacle_loss_torch , get_signed_dist_field


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable





traj_length = 20

n_links = 2 
obstacles = [ [0 , -0.6-0.43 , 0 , 5 , 0.1 , 5 ] , [-0.2 , -0.3-0.43 , 0 , 0.1 , 0.6 , 5 ]  ]
link_lengths=np.array([1,1]  )
target_pos = np.array([ -0.55 , -0.8 ,1 ] )
joint_angles_init  = np.array([[ 0.5,0.3,0.1  ] , [0.1,-0.6,1]] )



init_traj = np.array([ joint_angles_init for _ in range(traj_length) ])




obstacle_points = Variable(torch.from_numpy(get_grid_pts( obstacles[1] ))).reshape(-1 , 3 , 1)


link_lengths = Variable(torch.from_numpy(np.array(link_lengths).astype("float32")),requires_grad=True ).float()

joint_angles_init = Variable(torch.from_numpy(np.array(joint_angles_init).astype("float32")),requires_grad=True ).float()

target_pos = Variable(torch.from_numpy(np.array(target_pos).astype("float32")),requires_grad=True ).reshape(3,1).float()


q_traj_varible = Variable( torch.from_numpy(init_traj)).float()
q_traj_varible = torch.nn.Parameter(q_traj_varible)


optimizer = optim.SGD( [q_traj_varible] , lr=0.01, momentum=0.9)
# optimizer = optim.Adam( [q_traj_varible]  )

loss_fn_l2  = nn.MSELoss()




for it in range( 500 ):

	optimizer.zero_grad()

	dist_loss = 0 
	traj_smooth_loss = [] 
	dist_loss_soft = 0 

	for i in range(traj_length-1 ):
		q0 = q_traj_varible[i]
		q1 = q_traj_varible[i+1]

		joint_pos = forward_kinematics_arm_torch( link_lengths , q0  )
		l1 , l2 = get_obstacle_loss_torch( joint_pos ,  obstacle_points ) 
		dist_loss += l1 
		dist_loss_soft += l2 


		ww =  (torch.sum(( q0 - q1 )**2))
		traj_smooth_loss  += [torch.clamp( ww  - 0.09 , 0 , 1000000000 )[None]]

		
	traj_smooth_loss = torch.max( torch.cat(traj_smooth_loss))

	joint_pos = forward_kinematics_arm_torch( link_lengths , q1  )
	l1 , l2 = get_obstacle_loss_torch( joint_pos ,  obstacle_points ) 
	dist_loss += l1 
	dist_loss_soft += l2 
	final_pos = joint_pos[-1]

	target_loss = loss_fn_l2( final_pos[None] , target_pos[None] ) 






	


	loss = 10*target_loss  +  traj_smooth_loss  +  dist_loss*10 + 0.01*dist_loss_soft  

	print ("loss", float(loss.data) , float(dist_loss.data) , float(dist_loss_soft.data) , float(target_loss.data)  , float(traj_smooth_loss.data )) 

	loss.backward()
	optimizer.step()

s = None


while True:

	n_links = 2 
	obstacles = [ [0 , -0.6-0.43 , 0 , 5 , 0.1 , 5 ] , [-0.2 , -0.3-0.43 , 0 , 0.1 , 0.6 , 5 ]  ]
	link_lengths=np.array([1,1]  )
	target_pos = np.array([ -0.55 , -0.8 ,1 ] )
	joint_angles_init  = np.array([[ 0.5,0.3,0.1  ] , [0.1,-0.6,1]] )


	s = render_arm(link_lengths , joint_angles_init , target_pos , obstacles=obstacles ,s=s )
	time.sleep(0.1)

	for i in range(traj_length):
		s = render_arm(link_lengths , q_traj_varible.detach().data.numpy()[i] , target_pos , obstacles=obstacles ,s=s)
		time.sleep(0.1)




