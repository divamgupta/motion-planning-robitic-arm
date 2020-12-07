import numpy as np
from lib3d import Scene , Cylinder2P , Sphere , Box 



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



def  forward_kinematics_arm( link_lengths , joint_angles  ):

	link_lengths = np.array(link_lengths)
	joint_angles = np.array(joint_angles)

	n_joints = len(link_lengths)


	prev_pos = np.array([ 0.0, 0.0 , 0.0 ]  )
	prev_rot = np.eye(3 )  

	joint_pos = []


	joint_pos.append( prev_pos )

	for i in range(n_joints):
		pitch = joint_angles[ i , 0 ]  
		yaw = joint_angles[ i , 1 ] 
		roll = joint_angles[ i , 2 ]  
		lengt = link_lengths[ i ]  


		#https://en.wikipedia.org/wiki/Rotation_matrix
		rotmZ = np.array([ [np.cos(yaw) , -1*np.sin(yaw)  , 0 ] ,
			 [ np.sin(yaw) , np.cos(yaw) , 0  ] ,
			 [0 , 0 , 1 ] ]  )


		rotmX = np.array([ [ 1 , 0 , 0  ] ,
			 [  0 , np.cos(roll) , -1*np.sin(roll) ] ,
			 [ 0 ,  np.sin(roll) , np.cos(roll )] ] ) 

		rotmY = np.array([ [ np.cos(pitch) , 0 , np.sin(pitch)   ] ,
			 [   0 , 1 , 0  ] ,
			 [ -1*np.sin(pitch) , 0 , np.cos(pitch)  ] ]  )


		rotmZYX = rotmZ@rotmY@rotmX  

		arm_vec = [ 0 , 0 ,lengt  ]  

		prev_rot = rotmZYX@prev_rot  

		arm_vec =  prev_rot@arm_vec   

		next_pos = arm_vec + prev_pos   
		joint_pos.append( next_pos   ) 

		prev_pos = next_pos  

	return joint_pos



def  forward_kinematics_arm_torch( link_lengths , joint_angles  ):


	start_point = Variable(torch.zeros(3,1) ,requires_grad=True )


	n_joints = link_lengths.size()[0]


	prev_pos = start_point
	prev_rot = Variable(torch.eye(3 )  ,requires_grad=True  )

	joint_pos = []


	joint_pos.append( prev_pos )

	zeroo  = torch.zeros(1)[0]
	onee = torch.ones(1)[0]

	for i in range(n_joints):
		pitch = joint_angles[ i , 0 ]  
		yaw = joint_angles[ i , 1 ] 
		roll = joint_angles[ i , 2 ]  
		lengt = link_lengths[ i ]  



		#https://en.wikipedia.org/wiki/Rotation_matrix
		rotmZ = torch.cat( [torch.cos(yaw)[None] , -1*torch.sin(yaw)[None]  , zeroo[None] 
			, torch.sin(yaw)[None] , torch.cos(yaw)[None] , zeroo[None]   
			, zeroo[None]  , zeroo[None]  , onee[None] ]    ).reshape(3,3)


		rotmX = torch.cat(  [ onee[None] , zeroo[None]  , zeroo[None]   ,
			zeroo[None]  , torch.cos(roll)[None] , -1*torch.sin(roll)[None] ,
			 zeroo[None]  ,  torch.sin(roll)[None] , torch.cos(roll )[None]]   ).reshape(3,3)

		rotmY = torch.cat(  [ torch.cos(pitch)[None] , zeroo[None]  , torch.sin(pitch)[None]   ,
		  zeroo[None]  , onee[None] , zeroo[None]   ,
		  -1*torch.sin(pitch)[None] , zeroo[None]  , torch.cos(pitch)[None]  ]   ).reshape(3,3)


		rotmZYX = rotmZ@rotmY@rotmX  

		arm_vec = torch.transpose(Variable(torch.tensor([[ 0 , 0 ,lengt  ]]  ) ,requires_grad=True  ) , 0, 1 )

		prev_rot = rotmZYX@prev_rot  

		arm_vec =  prev_rot@arm_vec  

		# print( arm_vec.shape , prev_pos.shape ) 

		next_pos = arm_vec + prev_pos   
		joint_pos.append( next_pos   ) 

		prev_pos = next_pos  

	return joint_pos




def render_arm( link_lengths=[1,1]  , 
	joint_angles=[[ 0.5,0.3,0.1  ] , [0.1,-0.6,1]] ,  
	target_pos = [ 1,1,1 ], obstacles=[] , s=None ):


	joints = forward_kinematics_arm( link_lengths=link_lengths , joint_angles=joint_angles )

	if s is None:
		s = Scene()
	s.objects= []


	for i in range(len(joints)-1):
		sp = Cylinder2P( joints[i] , joints[i+1 ] , 0.05 , color=(1,0,0) )
		sp2 = Sphere( *list(joints[i]) ,0.05 , color=(1,0,0) )
		s.add_object( sp )
		s.add_object( sp2 )

	for ob in obstacles:
		s.add_object(Box(*ob ,   color=(0,1,1) ) ) 



	t =  Sphere( *list(target_pos ) ,0.1 , color=(1,1,0) ) 
	s.add_object(t)
	s.render()

	return s 




if __name__ == "__main__":
	link_lengths=Variable(torch.tensor([1,1])).float()
	joint_angles=Variable(torch.tensor([[ 0.5,0.3,0.1  ] , [0.1,-0.6,1]])).float()

	print( forward_kinematics_arm_torch( link_lengths , joint_angles  ) )

	print("-------")
	link_lengths= ([1,1]) 
	joint_angles= [[ 0.5,0.3,0.1  ] , [0.1,-0.6,1]] 
	print( forward_kinematics_arm(link_lengths , joint_angles) )

# if __name__ == "__main__":
# 	render_arm()
# 	input("press enter to exittttt")





