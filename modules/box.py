import numpy as np
from numpy.linalg import lstsq as optimizer
from scipy.spatial.transform import Rotation as rotation_util

median_object_size_map = {
    'Cyclist': (1.76, 1.75, 0.6),
    'Van': (4.98, 2.13, 1.88),
    'Tram': (14.66, 3.61, 2.6),
    'Car': (0.25, 0.13, 0.15),
    'Misc': (2.52, 1.65, 1.51),
    'Pedestrian': (0.88, 1.77, 0.65),
    'Truck': (10.81, 3.34, 2.63),
    'Person_sitting': (0.75, 1.26, 0.59),
    # 'DontCare': (-1.0, -1.0, -1.0)
}

EDGES = (
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
)

FACES = np.array([
    [5, 6, 8, 7],  # +x on yz plane
    [1, 3, 4, 2],  # -x on yz plane
    [3, 7, 8, 4],  # +y on xz plane = top
    [1, 2, 6, 5],  # -y on xz plane
    [2, 4, 8, 6],  # +z on xy plane = front
    [1, 5, 7, 3],  # -z on xy plane
])

UNIT_BOX = np.asarray([
    [0., 0., 0.],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
])

NUM_KEYPOINTS = 9
FRONT_FACE_ID = 4
TOP_FACE_ID = 2
def box_encoding(cls_labels, points_xyz, boxes_3d,
    label_map):
    encoded_boxes_3d = np.copy(boxes_3d)
    num_classes = boxes_3d.shape[1]
    points_xyz = np.expand_dims(points_xyz, axis=1)
    points_xyz = np.tile(points_xyz, (1, num_classes, 1))
    encoded_boxes_3d[:, :, 0] = boxes_3d[:, :, 0] - points_xyz[:, :, 0]
    encoded_boxes_3d[:, :, 1] = boxes_3d[:, :, 1] - points_xyz[:, :, 1]
    encoded_boxes_3d[:, :, 2] = boxes_3d[:, :, 2] - points_xyz[:, :, 2]
    for cls_name in label_map:
        if cls_name == "Background" or cls_name == "DontCare":
            continue
        cls_label = label_map[cls_name]
        l, h, w = median_object_size_map[cls_name]

        mask = cls_labels[:, 0] == cls_label
        encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/l
        encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/h
        encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/w
        encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
        encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
        encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
        encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6] #qw
        encoded_boxes_3d[mask, 0, 7] = boxes_3d[mask, 0, 7] #qx
        encoded_boxes_3d[mask, 0, 8] = boxes_3d[mask, 0, 8] #qy
        encoded_boxes_3d[mask, 0, 9] = boxes_3d[mask, 0, 9] #qz

    return encoded_boxes_3d

def box_decoding(cls_labels, points_xyz, encoded_boxes,
    label_map):
    decoded_boxes_3d = np.copy(encoded_boxes)
    for cls_name in label_map:
        if cls_name == "Background" or cls_name == "DontCare":
            continue
        cls_label = label_map[cls_name]
        l, h, w = median_object_size_map[cls_name]

        mask = cls_labels[:, 0] == cls_label
        decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*l
        decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
        decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*w
        decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
        decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
        decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
        decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6] #qw
        decoded_boxes_3d[mask, 0, 7] = encoded_boxes[mask, 0, 7] #qx
        decoded_boxes_3d[mask, 0, 8] = encoded_boxes[mask, 0, 8] #qy
        decoded_boxes_3d[mask, 0, 9] = encoded_boxes[mask, 0, 9] #qz

    # offset
    num_classes = encoded_boxes.shape[1]
    points_xyz = np.expand_dims(points_xyz, axis=1)
    points_xyz = np.tile(points_xyz, (1, num_classes, 1))
    decoded_boxes_3d[:, :, 0] = decoded_boxes_3d[:, :, 0] + points_xyz[:, :, 0]
    decoded_boxes_3d[:, :, 1] = decoded_boxes_3d[:, :, 1] + points_xyz[:, :, 1]
    decoded_boxes_3d[:, :, 2] = decoded_boxes_3d[:, :, 2] + points_xyz[:, :, 2]
    return decoded_boxes_3d
    
    
def classaware_all_class_box_encoding(cls_labels, points_xyz, boxes_3d,
    label_map):
    encoded_boxes_3d = np.copy(boxes_3d)
    num_classes = boxes_3d.shape[1]
    points_xyz = np.expand_dims(points_xyz, axis=1)
    points_xyz = np.tile(points_xyz, (1, num_classes, 1))
    encoded_boxes_3d[:, :, 0] = boxes_3d[:, :, 0] - points_xyz[:, :, 0]
    encoded_boxes_3d[:, :, 1] = boxes_3d[:, :, 1] - points_xyz[:, :, 1]
    encoded_boxes_3d[:, :, 2] = boxes_3d[:, :, 2] - points_xyz[:, :, 2]
    for cls_name in label_map:
        if cls_name == "Background" or cls_name == "DontCare":
            continue
        cls_label = label_map[cls_name]
        l, h, w = median_object_size_map[cls_name]

        # Car
        mask = cls_labels[:, 0] == cls_label
        encoded_boxes_3d[mask, 0, 0] = encoded_boxes_3d[mask, 0, 0]/l
        encoded_boxes_3d[mask, 0, 1] = encoded_boxes_3d[mask, 0, 1]/h
        encoded_boxes_3d[mask, 0, 2] = encoded_boxes_3d[mask, 0, 2]/w
        encoded_boxes_3d[mask, 0, 3] = np.log(boxes_3d[mask, 0, 3]/l)
        encoded_boxes_3d[mask, 0, 4] = np.log(boxes_3d[mask, 0, 4]/h)
        encoded_boxes_3d[mask, 0, 5] = np.log(boxes_3d[mask, 0, 5]/w)
        encoded_boxes_3d[mask, 0, 6] = boxes_3d[mask, 0, 6] #qw
        encoded_boxes_3d[mask, 0, 7] = boxes_3d[mask, 0, 7] #qx
        encoded_boxes_3d[mask, 0, 8] = boxes_3d[mask, 0, 8] #qy
        encoded_boxes_3d[mask, 0, 9] = boxes_3d[mask, 0, 9] #qz       

    return encoded_boxes_3d

def classaware_all_class_box_decoding(cls_labels, points_xyz, encoded_boxes,
    label_map):
    decoded_boxes_3d = np.copy(encoded_boxes)
    for cls_name in label_map:
        if cls_name == "Background" or cls_name == "DontCare":
            continue
        cls_label = label_map[cls_name]
        l, h, w = median_object_size_map[cls_name]

        # Car
        mask = cls_labels[:, 0] == cls_label
        decoded_boxes_3d[mask, 0, 0] = encoded_boxes[mask, 0, 0]*l
        decoded_boxes_3d[mask, 0, 1] = encoded_boxes[mask, 0, 1]*h
        decoded_boxes_3d[mask, 0, 2] = encoded_boxes[mask, 0, 2]*w
        decoded_boxes_3d[mask, 0, 3] = np.exp(encoded_boxes[mask, 0, 3])*l
        decoded_boxes_3d[mask, 0, 4] = np.exp(encoded_boxes[mask, 0, 4])*h
        decoded_boxes_3d[mask, 0, 5] = np.exp(encoded_boxes[mask, 0, 5])*w
        decoded_boxes_3d[mask, 0, 6] = encoded_boxes[mask, 0, 6] #qw
        decoded_boxes_3d[mask, 0, 7] = encoded_boxes[mask, 0, 7] #qx
        decoded_boxes_3d[mask, 0, 8] = encoded_boxes[mask, 0, 8] #qy
        decoded_boxes_3d[mask, 0, 9] = encoded_boxes[mask, 0, 9] #qz
                               
    # offset
    num_classes = encoded_boxes.shape[1]
    points_xyz = np.expand_dims(points_xyz, axis=1)
    points_xyz = np.tile(points_xyz, (1, num_classes, 1))
    decoded_boxes_3d[:, :, 0] = decoded_boxes_3d[:, :, 0] + points_xyz[:, :, 0]
    decoded_boxes_3d[:, :, 1] = decoded_boxes_3d[:, :, 1] + points_xyz[:, :, 1]
    decoded_boxes_3d[:, :, 2] = decoded_boxes_3d[:, :, 2] + points_xyz[:, :, 2]
    return decoded_boxes_3d


def get_box_encoding_fn(encoding_method_name):
    encoding_method_dict = {
        'classaware_all_class_box_encoding':classaware_all_class_box_encoding,
    }
    return encoding_method_dict[encoding_method_name]

def get_box_decoding_fn(encoding_method_name):
    decoding_method_dict = {
        'classaware_all_class_box_encoding': classaware_all_class_box_decoding,
    }
    return decoding_method_dict[encoding_method_name]

def get_encoding_len(encoding_method_name):
    encoding_len_dict = {
        'classaware_all_class_box_encoding': 10  # [ x,y,z,l,w,h,qw,qx,qy,qz]
    }
    return encoding_len_dict[encoding_method_name]
    


class Box(object):

  def __init__(self, vertices=None):
    if vertices is None:
      vertices = self.scaled_axis_aligned_vertices(np.array([1., 1., 1.]))

    self._vertices = vertices
    self._rotation = None
    self._translation = None
    self._scale = None
    self._transformation = None
    self._volume = None

  @classmethod
  def from_transformation(cls, rotation, translation, scale):
    if rotation.size != 3 and rotation.size != 9:
      raise ValueError('Unsupported rotation, only 3x1 euler angles or 3x3 ' +
                       'rotation matrices are supported. ' + rotation)
    if rotation.size == 3:
      rotation = rotation_util.from_rotvec(rotation.tolist()).as_dcm()
    scaled_identity_box = cls.scaled_axis_aligned_vertices(scale)
    vertices = np.zeros((NUM_KEYPOINTS, 3))
    for i in range(NUM_KEYPOINTS):
      vertices[i, :] = np.matmul(
          rotation, scaled_identity_box[i, :]) + translation.flatten()
    return cls(vertices=vertices)

  def __repr__(self):
    representation = 'Box: '
    for i in range(NUM_KEYPOINTS):
      representation += '[{0}: {1}, {2}, {3}]'.format(i, self.vertices[i, 0],
                                                      self.vertices[i, 1],
                                                      self.vertices[i, 2])
    return representation

  def __len__(self):
    return NUM_KEYPOINTS

  def __name__(self):
    return 'Box'

  def apply_transformation(self, transformation):
    if transformation.shape != (4, 4):
      raise ValueError('Transformation should be a 4x4 matrix.')

    new_rotation = np.matmul(transformation[:3, :3], self.rotation)
    new_translation = transformation[:3, 3] + (
        np.matmul(transformation[:3, :3], self.translation))
    return Box.from_transformation(new_rotation, new_translation, self.scale)

  @classmethod
  def scaled_axis_aligned_vertices(cls, scale):
    w = scale[0] / 2.
    h = scale[1] / 2.
    d = scale[2] / 2.

    aabb = np.array([[0., 0., 0.], [-w, -h, -d], [-w, -h, +d], [-w, +h, -d],
                     [-w, +h, +d], [+w, -h, -d], [+w, -h, +d], [+w, +h, -d],
                     [+w, +h, +d]])
    return aabb

  @classmethod
  def fit(cls, vertices):
    orientation = np.identity(3)
    translation = np.zeros((3, 1))
    scale = np.zeros(3)

    for axis in range(3):
      for edge_id in range(4):
        begin, end = EDGES[axis * 4 + edge_id]
        scale[axis] += np.linalg.norm(vertices[begin, :] - vertices[end, :])
      scale[axis] /= 4.

    x = cls.scaled_axis_aligned_vertices(scale)
    system = np.concatenate((x, np.ones((NUM_KEYPOINTS, 1))), axis=1)
    solution, _, _, _ = optimizer(system, vertices, rcond=None)
    orientation = solution[:3, :3].T
    translation = solution[3, :3]
    return orientation, translation, scale

  def inside(self, point):
    inv_trans = np.linalg.inv(self.transformation)
    scale = self.scale
    point_w = np.matmul(inv_trans[:3, :3], point) + inv_trans[:3, 3]
    for i in range(3):
      if abs(point_w[i]) > scale[i] / 2.:
        return False
    return True

  def sample(self):
    point = np.random.uniform(-0.5, 0.5, 3) * self.scale
    point = np.matmul(self.rotation, point) + self.translation
    return point

  @property
  def vertices(self):
    return self._vertices

  @property
  def rotation(self):
    if self._rotation is None:
      self._rotation, self._translation, self._scale = self.fit(self._vertices)
    return self._rotation

  @property
  def translation(self):
    if self._translation is None:
      self._rotation, self._translation, self._scale = self.fit(self._vertices)
    return self._translation

  @property
  def scale(self):
    if self._scale is None:
      self._rotation, self._translation, self._scale = self.fit(self._vertices)
    return self._scale

  @property
  def volume(self):
    if self._volume is None:
      i = self._vertices[2, :] - self._vertices[1, :]
      j = self._vertices[3, :] - self._vertices[1, :]
      k = self._vertices[5, :] - self._vertices[1, :]
      sys = np.array([i, j, k])
      self._volume = abs(np.linalg.det(sys))
    return self._volume

  @property
  def transformation(self):
    if self._rotation is None:
      self._rotation, self._translation, self._scale = self.fit(self._vertices)
    if self._transformation is None:
      self._transformation = np.identity(4)
      self._transformation[:3, :3] = self._rotation
      self._transformation[:3, 3] = self._translation
    return self._transformation

  def get_ground_plane(self, gravity_axis=1):
    gravity = np.zeros(3)
    gravity[gravity_axis] = 1

    def get_face_normal(face, center):
      v1 = self.vertices[face[0], :] - center
      v2 = self.vertices[face[1], :] - center
      normal = np.cross(v1, v2)
      return normal

    def get_face_center(face):
      center = np.zeros(3)
      for vertex in face:
        center += self.vertices[vertex, :]
      center /= len(face)
      return center

    ground_plane_id = 0
    ground_plane_error = 10.

    for i in [0, 2, 4]:
      face = FACES[i, :]
      center = get_face_center(face)
      normal = get_face_normal(face, center)
      w = np.cross(gravity, normal)
      w_sq_norm = np.linalg.norm(w)
      if w_sq_norm < ground_plane_error:
        ground_plane_error = w_sq_norm
        ground_plane_id = i

    face = FACES[ground_plane_id, :]
    center = get_face_center(face)
    normal = get_face_normal(face, center)

    parallel_face_id = ground_plane_id + 1
    parallel_face = FACES[parallel_face_id]
    parallel_face_center = get_face_center(parallel_face)
    parallel_face_normal = get_face_normal(parallel_face, parallel_face_center)
    if parallel_face_center[gravity_axis] < center[gravity_axis]:
      center = parallel_face_center
      normal = parallel_face_normal
    return center, normal
