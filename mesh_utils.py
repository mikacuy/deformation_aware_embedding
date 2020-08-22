import os
import sys
import numpy as np
import h5py
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'retrieval/utils'))
import pc_util
import objloader

def face_areas(v1, v2, v3) :
    return 0.5 * np.linalg.norm( np.cross( v2-v1, v3-v1 ), axis=1)

# def mesh_to_pc(filename, output_filename, NUM_POINTS= 15000):
def mesh_to_pc(V, F, num_points= 2000):

	# V, F, VT, FT, VN, FN, face_mat, kdmap = objloader.LoadSimpleOBJ_manifold(filename)

	vertices = V
	faces = F
	# print("Number of vertices: "+str(V.shape[0]))
	# print("Number of faces: "+str(F.shape[0]))

	vertices = pd.DataFrame(vertices, columns = ['x', 'y', 'z'])

	faces = np.array(faces)
	faces = pd.DataFrame(faces, columns = ['v1', 'v2', 'v3'])

	#Construct dict with vertices and faces of mesh for sampling point cloud later where each value is a pandas dict
	mesh = {"vertices": vertices, "faces": faces}

	###Faster
	mesh_points_xyz = mesh["vertices"][["x","y","z"]].values
	v1_xyz = mesh_points_xyz[mesh["faces"]["v1"]]
	v2_xyz = mesh_points_xyz[mesh["faces"]["v2"]]
	v3_xyz = mesh_points_xyz[mesh["faces"]["v3"]]
	areas = face_areas(v1_xyz, v2_xyz, v3_xyz)
	probs = areas/np.sum(areas)

	#faces are selected weighted by their areas
	face_indices = np.random.choice(range(len(areas)), size= num_points, p=probs)

	selected_v1_xyz = v1_xyz[face_indices]
	selected_v2_xyz = v2_xyz[face_indices]
	selected_v3_xyz = v3_xyz[face_indices]

	##Barycentric coordinates
	u = np.random.rand(num_points, 1)
	v = np.random.rand(num_points, 1)
	is_prob = (u+v) >1
	u[is_prob] = 1 - u[is_prob]
	v[is_prob] = 1 - v[is_prob]
	w = 1 - u - v

	sampled_pc = selected_v1_xyz*u + selected_v2_xyz*v + selected_v3_xyz*w
	# print(sampled_pc.shape)
	# pc_util.write_ply(sampled_pc, output_filename)
	return sampled_pc


# mesh_to_pc("test3.obj", "test3.ply", 2000)