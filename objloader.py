import numpy as np
import skimage.io as sio

def LoadOff(model_path):
	lines = [l.strip() for l in open(model_path)]
	words = [int(i) for i in lines[1].split(' ')]
	vn = words[0]
	fn = words[1]
	vertices = np.zeros((vn, 3), dtype='float32')
	faces = np.zeros((fn, 3), dtype='int32')
	for i in range(2, 2 + vn):
		vertices[i-2] = [float(w) for w in lines[i].split(' ')]
	for i in range(2 + vn, 2 + vn + fn):
		digits = [int(w) for w in lines[i].split(' ')]
		if digits[0] != 3:
			print('cannot parse...')
			exit(0)
		faces[i-2-vn] = digits[1:]
	return vertices, faces
	
def LoadTextureOBJ(model_path):
  vertices = []
  vertex_textures = []
  vertex_normals = []
  faces = []
  face_mat = []
  face_textures = []
  face_normals = []
  lines = [l.strip() for l in open(model_path)]
  materials = {}
  kdmap = []
  mat_idx = -1
  filename = model_path.split('/')[-1]
  file_dir = model_path[:-len(filename)]

  for l in lines:
    words = [w for w in l.split(' ') if w != '']

    if len(words) == 0:
      continue

    if words[0] == 'mtllib':
      model_file = model_path.split('/')[-1]
      mtl_file = model_path[:-len(model_file)] + words[1]
      mt_lines = [l.strip() for l in open(mtl_file) if l != '']
      for mt_l in mt_lines:
        mt_words = [w for w in mt_l.split(' ') if w != '']
        if (len(mt_words) == 0):
          continue
        if mt_words[0] == 'newmtl':
          key = mt_words[1]
          materials[key] = np.array([[[0,0,0]]]).astype('uint8')
        if mt_words[0] == 'Kd':
          materials[key] = np.array([[[float(mt_words[1])*255, float(mt_words[2])*255, float(mt_words[3])*255]]]).astype('uint8')
        if mt_words[0] == 'map_Kd':
          if mt_words[1][0] != '/':
            img = sio.imread(file_dir + mt_words[1])
          else:
            img = sio.imread(mt_words[1])
          if len(img.shape) == 2:
            img = np.dstack((img,img,img))
          elif img.shape[2] >= 4:
            img = img[:,:,0:3]
          materials[key] = img

    if words[0] == 'v':
      vertices.append([float(words[1]), float(words[2]), float(words[3])])
    if words[0] == 'vt':
      vertex_textures.append([float(words[1]), float(words[2])])
    if words[0] == 'vn':
      vertex_normals.append([float(words[1]), float(words[2]), float(words[3])])

    if words[0] == 'usemtl':
      mat_idx = len(kdmap)

      kdmap.append(materials[words[1]])

    if words[0] == 'f':
      print(words)

      f = []
      ft = []
      fn = []
      for j in range(3):
        w = words[j + 1].split('/')[0]
        wt = words[j + 1].split('/')[1]
        wn = words[j + 1].split('/')[2]
        f.append(int(w) - 1)
        ft.append(int(wt) - 1)
        fn.append(int(wn) - 1)
      faces.append(f)
      face_textures.append(ft)
      face_normals.append(fn)
      face_mat.append(mat_idx)
  F = np.array(faces, dtype='int32')
  V = np.array(vertices, dtype='float32')
  V = (V * 0.5).astype('float32')
  VN = np.array(vertex_normals, dtype='float32')
  VT = np.array(vertex_textures, dtype='float32')
  FT = np.array(face_textures, dtype='int32')
  FN = np.array(face_normals, dtype='int32')
  face_mat = np.array(face_mat, dtype='int32')

  return V, F, VT, FT, VN, FN, face_mat, kdmap

def LoadSimpleOBJ(model_path):
  vertices = []
  vertex_textures = []
  vertex_normals = []
  faces = []
  face_mat = []
  face_textures = []
  face_normals = []
  lines = [l.strip() for l in open(model_path)]

  mat_idx = -1
  filename = model_path.split('/')[-1]
  file_dir = model_path[:-len(filename)]

  for l in lines:
    words = [w for w in l.split(' ') if w != '']

    if len(words) == 0:
      continue

    if words[0] == 'v':
      vertices.append([float(words[1]), float(words[2]), float(words[3])])
    if words[0] == 'vt':
      vertex_textures.append([float(words[1]), float(words[2])])
    if words[0] == 'vn':
      vertex_normals.append([float(words[1]), float(words[2]), float(words[3])])

    if words[0] == 'f':

      f = []
      ft = []
      fn = []
      for j in range(3):
        w = words[j + 1].split('/')[0]
        # wt = words[j + 1].split('/')[1]
        wn = words[j + 1].split('/')[2]
        f.append(int(w) - 1)
        # ft.append(int(wt) - 1)
        fn.append(int(wn) - 1)
      faces.append(f)
      # face_textures.append(ft)
      face_normals.append(fn)


  F = np.array(faces, dtype='int32')
  V = np.array(vertices, dtype='float32')
  V = (V * 0.5).astype('float32')
  VN = np.array(vertex_normals, dtype='float32')
  VT = np.array(vertex_textures, dtype='float32')
  FT = np.array(face_textures, dtype='int32')
  FN = np.array(face_normals, dtype='int32')
  face_mat = np.array(face_mat, dtype='int32')

  return V, F, VT, FT, VN, FN, None, None

def LoadSimpleOBJ_manifold(model_path):
  vertices = []
  vertex_textures = []
  vertex_normals = []
  faces = []
  face_mat = []
  face_textures = []
  face_normals = []
  f = open(model_path, "r")
  lines = [l.strip() for l in f]
  f.close()

  mat_idx = -1
  filename = model_path.split('/')[-1]
  file_dir = model_path[:-len(filename)]

  for l in lines:
    words = [w for w in l.split(' ') if w != '']

    if len(words) == 0:
      continue

    if words[0] == 'v':
      vertices.append([float(words[1]), float(words[2]), float(words[3])])
    if words[0] == 'vt':
      vertex_textures.append([float(words[1]), float(words[2])])
    if words[0] == 'vn':
      vertex_normals.append([float(words[1]), float(words[2]), float(words[3])])

    if words[0] == 'f':

      f = []
      ft = []
      fn = []
      for j in range(3):
        w = words[j + 1]
        # wt = words[j + 1].split('/')[1]
        f.append(int(w) - 1)
        # ft.append(int(wt) - 1)
        # fn.append(int(wn) - 1)
      faces.append(f)
      # face_textures.append(ft)
      # face_normals.append(fn)


  F = np.array(faces, dtype='int32')
  V = np.array(vertices, dtype='float32')
  V = (V * 0.5).astype('float32')
  VN = np.array(vertex_normals, dtype='float32')
  VT = np.array(vertex_textures, dtype='float32')
  FT = np.array(face_textures, dtype='int32')
  FN = np.array(face_normals, dtype='int32')
  face_mat = np.array(face_mat, dtype='int32')

  return V, F, VT, FT, VN, FN, None, None
