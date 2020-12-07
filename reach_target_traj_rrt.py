
# just so that you know this code is written really reallly bad and i wrote it half asleep  . half variables are global and stuff 
# but it works .. so yeah

import numpy as np 
from arm import render_arm , forward_kinematics_arm 
from reach_target_optimization import final_state_arm_optimze , get_obstacle_loss , get_grid_pts , get_signed_dist_field
import random
from tqdm import tqdm 
import time 

from scipy.spatial.transform import Rotation 
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

all_verts = np.array([])
graph = {}

def get_nearest(v ):
	dists = np.linalg.norm(all_verts - v , axis=1 )
	idd = np.argmin( dists) 
	return idd , all_verts[ idd]


def new_vert( v ):
	global all_verts 
	if len(all_verts) == 0:
		all_verts = np.array([v])
	else:
		all_verts = np.vstack((all_verts,  v  ))

	node_id = len( all_verts ) - 1  
	graph[node_id] = []

	return 	node_id


def is_colliding( q_start , q_end , n_inter=0  ) : # check if  the arm collides from q1 to q0 
	
	if n_inter == 0:
		return False  

	# for alp in np.linspace( 0 , 1 , n_inter ):
	# 	q = q_start*( 1-alp ) + q_end*alp 
	for q in interp_q( q_start , q_end , n_inter):
		joint_pos = forward_kinematics_arm( link_lengths , q.reshape((-1 , 3 )) )
		if get_obstacle_loss( joint_pos ,  dists_field ) > 0.001 :
			return True 
	return False 


def dist_in_q( q1  , q2 ):

	d1 =  np.linalg.norm( q1 - q2 ) 

	# # 1 
	# q1 = q1.reshape((-1,3))
	# q2 = q2.reshape((-1,3))

	# pts_1 = np.array(forward_kinematics_arm(link_lengths,  q1 ))
	# pts_2 = np.array(forward_kinematics_arm( link_lengths ,  q2 ))

	# d2 =  np.linalg.norm( pts_2 - pts_1 )

	return d1 # + d2 

	# # 2 
	# dist=0
	# for t1 , t2 in zip( q1 , q2 ):
	# 	quat_1 = Rotation.from_euler('xyz', t1 ).as_quat()
	# 	quat_2 = Rotation.from_euler('xyz', t2 ).as_quat()
	# 	angle = 2*math.acos(q[0] )
	# 	min( angle , 2*math.pi - angle )



def interp_q( q1 , q2 , n_inter):
	q1 = q1.reshape((-1 , 3 ))
	q2 = q2.reshape((-1 , 3 ))

	for alp in np.linspace( 0 , 1 , n_inter ):

		q_new = ( 1-alp )*q1 + alp*q2

		# q_new = np.zeros_like( q1 )

		# for i , ( t1 , t2) in enumerate( zip( q1 , q2 )):
		# 	quat_1 = Rotation.from_euler('xyz', t1 ).as_quat()
		# 	quat_2 = Rotation.from_euler('xyz', t2 ).as_quat() 
		# 	q_new[i] =  R.from_quat( ( 1-alp )*quat_1 + alp*quat_2 ).as_euler("xyz")

		yield q_new


# def interp_q( q1 , q2 , n_inter):
# 	q1 = q1.reshape((-1 , 3 ))
# 	q2 = q2.reshape((-1 , 3 ))

# 	n_len = len( q1 )

# 	rots = []

# 	for i in range(n_len):
# 		key_rots  = R.from_euler('xyz',[ q1[i] , q2[i]] )
# 		key_times = [ 0 , 1 ]
# 		slerp = Slerp(key_times, key_rots)
# 		times = np.linspace( 0 , 1 , n_inter )
# 		interp_rots = slerp(times)
# 		interp_rots = interp_rots.as_euler('xyz' ) 
# 		rots.append( interp_rots )

# 	for j in range( n_inter ):
# 		q_new = np.zeros_like( q1 )
# 		for i in range(n_len):
# 			q_new[i] = rots[i][j]
# 		yield q_new 
		








def get_random_q():
	
	x = np.random.rand(n_links , 3)*2*np.pi  
	joint_pos = forward_kinematics_arm( link_lengths , x.reshape((-1 , 3 )) )


	if get_obstacle_loss( joint_pos ,  dists_field ) > 0.01 :
		# print("colide")
		return get_random_q()
	else:
		return x



def add_vert( q ):
	ner_id , _ = get_nearest( q )
	n_id = new_vert( q )
	

	if not is_colliding( all_verts[n_id] , all_verts[ner_id]):
		graph[n_id ].append( ner_id)
		graph[ ner_id].append(n_id )


def do_bfs( start  ):
	visited = {}
	tree = {}
	queue = []

	queue.append( start )
	visited[start] = True 

	while len(queue) > 0:
		el = queue.pop( 0 )

		tree[el] = []
		nebors = graph[ el ]
		for n in nebors:
			if not n in visited:
				queue.append( n )
				tree[el].append( n )
				visited[n] = True 
	return tree , visited.keys()



def compute_dists_tree( tree , root=None ): # for each node compute the dist to its parent node 
	global dists 

	if root == 0:
		dists = {}
		dists[0] = 0 

	for el in tree[root] :
		dists[ el ] = dist_in_q( all_verts[ root ] , all_verts[el ] )
		compute_dists_tree( tree , el )

	return dists 


def shortest_path_tree( tree , dists , start , goal  ):

	min_d = 1000000000
	min_path = [ ]

	if start == goal:
		return 0 , [ goal ]

	for el in tree[ start ]:


		if len(tree[el]) > 0:
			c_dist , c_path = shortest_path_tree( tree , dists , el  , goal )
			c_dist += dists[ el ]
			c_path = [start] + c_path 
		else:
			if el == goal:
				c_dist = dists[ el ]
				c_path = [ start , el ]
			else:
				c_dist = 1000000000 # inf dist if the node is not conencted with the goal 
				c_path = [ start , el ]

		if c_dist < min_d:
			min_d = c_dist
			min_path = c_path 

	return min_d , min_path





s = None
n_links = 2 
obstacles = [ [0 , -0.6-0.43 , 0 , 5 , 0.1 , 5 ] , [-0.2 , -0.3-0.43 , 0 , 0.1 , 0.6 , 5 ]  ]
link_lengths=np.array([1,1]  )
target_pos = np.array([ -0.55 , -0.8 ,1 ] )
joint_angles_init  = np.array([[ 0.5,0.3,0.1  ] , [0.1,-0.6,1]] )

obs_pts = get_grid_pts( obstacles[1] )
dists_field = get_signed_dist_field( obs_pts )


n_start = new_vert( joint_angles_init.flatten() )

for _ in tqdm(range( 10000 )):
	rand_q  = get_random_q()
	add_vert( rand_q.flatten() )




# s = render_arm(link_lengths , joint_angles_init , target_pos , obstacles=obstacles ,s=s )
# input()



bfs_tree , bfs_nodes = do_bfs( n_start ) 

tree_dists = compute_dists_tree( bfs_tree  , 0)
# print ( tree_dists )


min_d = 10000000000
best_goal = None 


for nn in bfs_nodes:
	angles = all_verts[nn].reshape((-1 , 3 ))
	joint_pos = forward_kinematics_arm( link_lengths , angles.reshape((-1 , 3 )) )
	final_pos = joint_pos[-1]
	dist_from_target  = np.sum( (final_pos -target_pos  )**2 )
	if dist_from_target < min_d:
		min_d = dist_from_target
		best_goal = nn  

_ , path = shortest_path_tree( bfs_tree , tree_dists , n_start , best_goal  )

# now create a dense graph using the nodes returned 
graph = {} # bad code i know :) 

for n_i in path:
	for n_j in path:
		if not is_colliding( all_verts[n_i] , all_verts[n_j] , 30  ):

			if not n_j in graph:
				graph[n_j] = []

			if not n_i in graph:
				graph[n_i] = []

			graph[n_i].append( n_j )
			graph[n_j].append( n_i )

bfs_tree , bfs_nodes = do_bfs( n_start ) 
tree_dists = compute_dists_tree( bfs_tree  , 0)

_ , path = shortest_path_tree( bfs_tree , tree_dists , n_start , best_goal  )


path_interp = []

for i in range( len(path) -1 ):
	q1 = all_verts[path[i]]
	q2 = all_verts[ path[i+1 ]]
	# for alp in np.linspace( 0 , 1 , 20 ):
	# 	q = q1*( 1-alp ) + q2 *alp 
	for q in interp_q( q1 , q2 , 20):
		path_interp.append( q )


print( bfs_tree )




print( path )

while True:
	for v in path_interp:
		aa = v.reshape((-1 ,3 ))
		s = render_arm(link_lengths , aa  , target_pos , obstacles=obstacles ,s=s )
		time.sleep(0.1)
	time.sleep( 1 )








# exit( )

