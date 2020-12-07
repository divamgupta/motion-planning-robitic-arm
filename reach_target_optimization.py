import numpy as np 
from scipy.optimize import leastsq

from arm import render_arm , forward_kinematics_arm , forward_kinematics_arm_torch 
import scipy

sdf_grid_step = 0.1
sdf_grid_N = len(np.arange( -1 , 1 , sdf_grid_step))



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable




OBSTACLE_LOSS = 1


def get_grid_pts( obs ):
	grid_step = 0.11
	
	x_ax = list(np.arange( obs[0]  ,  obs[0]-obs[3] , -grid_step)[1:]) + list(np.arange( obs[0]  ,  obs[0]+obs[3] , grid_step))

	y_ax = list(np.arange( obs[1]  ,  obs[1]+obs[4] , grid_step)) + list(np.arange( obs[1]  ,  obs[1]-obs[4] , -grid_step)[1:])

	z_ax = list(np.arange( obs[2]  ,  obs[2]+obs[5] , grid_step)) + list(np.arange( obs[2]  ,  obs[2]-obs[5] , -grid_step)[1:])
	pts = []
	for x in x_ax:
		for y in y_ax:
			for z in z_ax:
				pts.append(np.array([x,y,z ]))
	return np.array(pts) 




def get_signed_dist_field( obstacle_points ):
	

	
	dists_map  = np.zeros((sdf_grid_N , sdf_grid_N , sdf_grid_N ))

	for xi ,  x in enumerate(np.arange( -1 , 1 , sdf_grid_step)):
		for yi , y in enumerate(np.arange(-1 , 1  , sdf_grid_step)):
			for zi ,  z in enumerate( np.arange(-1 , 1 , sdf_grid_step)):
				tt = np.array([x, y , z ])
				dists = np.sqrt( np.sum((obstacle_points - tt )**2 , -1 ))
				min_d = np.min(dists )
				dists_map[xi , yi , zi ] = min_d 
	return dists_map



def interpolate_joint_pts( joints  ):
	joints = np.array( joints )
	pts_ret = []
	for i in range(len(joints)-1):
		p1 , p2 = joints[i] , joints[i+1 ]
		for alp in np.arange( 0 , 1 , 0.1 ):
			pts_ret.append( p1 + alp*(p2-p1))
	return np.array( pts_ret )






def interpolate_joint_pts_torch( joints  ):

	pts_ret = []
	for i in range(len(joints)-1):
		p1 , p2 = joints[i] , joints[i+1 ]
		for alp in np.arange( 0 , 1 , 0.1 ):
			pts_ret.append( (p1 + alp*(p2-p1))[None] )
	return  torch.cat(pts_ret , 0 ) 




def get_dists_from_sdf( points , dists_field  ):
	points_quant = ((points + 1 )/sdf_grid_step).astype(int )
	points_quant = np.clip( points_quant , 0 , sdf_grid_N  - 1  )
	return dists_field[ points_quant[: , 0 ] , points_quant[: , 1 ] , points_quant[: , 2 ]   ]




def get_obstacle_loss( joint_pos ,  dists_field ):
	joint_pos_interpol = interpolate_joint_pts( joint_pos )

	min_d = np.min( get_dists_from_sdf( joint_pos_interpol , dists_field  ) ) 

	obstacle_loss = max( 0 ,  0.1 -  min_d )
	return obstacle_loss 



def reach_target_final_state_loss_fn( x , link_lengths , target_pos  , dists_field ):
	joint_pos = forward_kinematics_arm( link_lengths , x.reshape((-1 , 3 )) )

	

	# print( obstacle_loss )
	final_pos = joint_pos[-1]
	target_loss = np.sum( (final_pos -target_pos  )**2 )

	obstacle_loss = get_obstacle_loss( joint_pos ,  dists_field )


	return  target_loss + 100*obstacle_loss*OBSTACLE_LOSS 




def final_state_arm_optimze(link_lengths ,joint_angles_init , target_pos ,  obstacles ):

	obs_pts = get_grid_pts( obstacles[1] )
	dists_field = get_signed_dist_field( obs_pts)


	x_init = joint_angles_init.flatten()
	x_opt = scipy.optimize.minimize(reach_target_final_state_loss_fn , x_init, args=( link_lengths , target_pos  , dists_field  ), method='SLSQP')
	x_opt = np.array(x_opt.x)

	joint_angles_final = x_opt.reshape((-1,3 )) 

	return joint_angles_final

	


# def get_dists_from_sdf_torch( points , dists_field  ):
# 	points_quant = ((points + 1 )/sdf_grid_step)
# 	points_quant = torch.clip( points_quant , 0 , sdf_grid_N  - 1  )
# 	return dists_field[ points_quant[: , 0 ] , points_quant[: , 1 ] , points_quant[: , 2 ]   ]




def get_obstacle_loss_torch( joint_pos ,  obstacle_points ):
	joint_pos_interpol = interpolate_joint_pts_torch( joint_pos )

	min_d = min_dist_from_pts( joint_pos_interpol , obstacle_points  )

	obstacle_loss = torch.clamp(   0.1 -  min_d , 0 , 1 )
	obstacle_loss_soft = torch.clamp(   0.4 -  min_d , 0 , 1 )


	return obstacle_loss , obstacle_loss_soft 



def min_dist_from_pts( robot_pts  , obstacle_points):
	assert robot_pts.shape[1:] == (3,1)
	assert obstacle_points.shape[1:] == (3,1)

	dists = torch.sqrt( torch.sum((obstacle_points[None ] - robot_pts[: , None ] )**2 , 2  ))
	# print( dists.shape , "yeh chokor hona chaiye")
	return torch.min( dists )


def final_state_arm_optimze_torch(link_lengths ,joint_angles_init , target_pos ,  obstacles ):

	

	link_lengths = Variable(torch.from_numpy(np.array(link_lengths).astype("float32")),requires_grad=True ).float()

	joint_angles_init = Variable(torch.from_numpy(np.array(joint_angles_init).astype("float32")),requires_grad=True ).float()


	target_pos = Variable(torch.from_numpy(np.array(target_pos).astype("float32")),requires_grad=True ).reshape(3,1).float()

	q_varible = torch.nn.Parameter(joint_angles_init.clone())

	optimizer = optim.SGD( [q_varible] , lr=0.01, momentum=0.9)

	loss_fn = nn.MSELoss()


	obstacle_points = Variable(torch.from_numpy(get_grid_pts( obstacles[1] ))).reshape(-1 , 3 , 1)

	print( obstacle_points.shape  , "obstacle_points/shpe")


	for it in range( 1000 ):

		optimizer.zero_grad()

		joint_pos = forward_kinematics_arm_torch( link_lengths , q_varible )

		final_pos = joint_pos[-1]

		target_loss = loss_fn( final_pos[None] , target_pos[None] ) 
		dist_loss = get_obstacle_loss_torch( joint_pos ,  obstacle_points )[0]


		loss = target_loss  + dist_loss*100*OBSTACLE_LOSS 

		print ("loss", float(loss.data) , float(dist_loss.data) , float(target_loss.data) ) 

		loss.backward()
		optimizer.step()


		

	return q_varible.data.numpy()






if __name__ == "__main__":
	
	obstacles = [ [0 , -0.6-0.43 , 0 , 5 , 0.1 , 5 ] , [-0.2 , -0.3-0.43 , 0 , 0.1 , 0.6 , 5 ]  ]
	link_lengths=np.array([1,1]  )
	target_pos = np.array([ -0.55 , -0.8 ,1 ] )
	joint_angles_init  =np.array([[ 0.5,0.3,0.1  ] , [0.1,-0.6,1]] )

	joint_angles_final = final_state_arm_optimze_torch(link_lengths ,joint_angles_init , target_pos ,  obstacles )

	render_arm(link_lengths , joint_angles_init , target_pos , obstacles=obstacles)

	input("Press enter")

	render_arm(link_lengths , joint_angles_final , target_pos , obstacles=obstacles )

	input("Press enter")




