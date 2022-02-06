import numpy as np
import scipy.spatial as sp

from modules.box import *

_PLANE_THICKNESS_EPSILON = 0.000001
_POINT_IN_FRONT_OF_PLANE = 1
_POINT_ON_PLANE = 0
_POINT_BEHIND_PLANE = -1


class IoU(object):

  def __init__(self, box1, box2):
    self._box1 = box1
    self._box2 = box2
    self._intersection_points = []

  def setBox(self, box1, box2):
    self._box1 = box1
    self._box2 = box2

  def iou(self):
    self._intersection_points = []
    self._compute_intersection_points(self._box1, self._box2)
    self._compute_intersection_points(self._box2, self._box1)
    if self._intersection_points:
      intersection_volume = sp.ConvexHull(self._intersection_points).volume
      box1_volume = self._box1.volume
      box2_volume = self._box2.volume
      union_volume = box1_volume + box2_volume - intersection_volume
      return intersection_volume / union_volume
    else:
      return 0.

  def iou_sampling(self, num_samples=10000):
    p1 = [self._box1.sample() for _ in range(num_samples)]
    p2 = [self._box2.sample() for _ in range(num_samples)]
    box1_volume = self._box1.volume
    box2_volume = self._box2.volume
    box1_intersection_estimate = 0
    box2_intersection_estimate = 0
    for point in p1:
      if self._box2.inside(point):
        box1_intersection_estimate += 1
    for point in p2:
      if self._box1.inside(point):
        box2_intersection_estimate += 1

    intersection_volume_estimate = (
        box1_volume * box1_intersection_estimate +
        box2_volume * box2_intersection_estimate) / 2.0
    union_volume_estimate = (box1_volume * num_samples + box2_volume *
                             num_samples) - intersection_volume_estimate
    iou_estimate = intersection_volume_estimate / union_volume_estimate
    return iou_estimate

  def _compute_intersection_points(self, box_src, box_template):
    inv_transform = np.linalg.inv(box_src.transformation)
    box_src_axis_aligned = box_src.apply_transformation(inv_transform)
    template_in_src_coord = box_template.apply_transformation(inv_transform)
    
    for face in range(len(FACES)):
      indices = FACES[face, :]
      poly = [template_in_src_coord.vertices[indices[i], :] for i in range(4)]
      clip = self.intersect_box_poly(box_src_axis_aligned, poly)
      for point in clip:
        point_w = np.matmul(box_src.rotation, point) + box_src.translation
        self._intersection_points.append(point_w)

    for point_id in range(NUM_KEYPOINTS):
      v = template_in_src_coord.vertices[point_id, :]
      if box_src_axis_aligned.inside(v):
        point_w = np.matmul(box_src.rotation, v) + box_src.translation
        self._intersection_points.append(point_w)

  def intersect_box_poly(self, box, poly):
    for axis in range(3):
      poly = self._clip_poly(poly, box.vertices[1, :], 1.0, axis)
      poly = self._clip_poly(poly, box.vertices[8, :], -1.0, axis)
    return poly

  def _clip_poly(self, poly, plane, normal, axis):
    result = []
    if len(poly) <= 1:
      return result

    poly_in_plane = True

    for i, current_poly_point in enumerate(poly):
      prev_poly_point = poly[(i + len(poly) - 1) % len(poly)]
      d1 = self._classify_point_to_plane(prev_poly_point, plane, normal, axis)
      d2 = self._classify_point_to_plane(current_poly_point, plane, normal,
                                         axis)
      if d2 == _POINT_BEHIND_PLANE:
        poly_in_plane = False
        if d1 == _POINT_IN_FRONT_OF_PLANE:
          intersection = self._intersect(plane, prev_poly_point,
                                         current_poly_point, axis)
          result.append(intersection)
        elif d1 == _POINT_ON_PLANE:
          if not result or (not np.array_equal(result[-1], prev_poly_point)):
            result.append(prev_poly_point)
      elif d2 == _POINT_IN_FRONT_OF_PLANE:
        poly_in_plane = False
        if d1 == _POINT_BEHIND_PLANE:
          intersection = self._intersect(plane, prev_poly_point,
                                         current_poly_point, axis)
          result.append(intersection)
        elif d1 == _POINT_ON_PLANE:
          if not result or (not np.array_equal(result[-1], prev_poly_point)):
            result.append(prev_poly_point)

        result.append(current_poly_point)
      else:
        if d1 != _POINT_ON_PLANE:
          result.append(current_poly_point)

    if poly_in_plane:
      return poly
    else:
      return result

  def _intersect(self, plane, prev_point, current_point, axis):
    alpha = (current_point[axis] - plane[axis]) / (
        current_point[axis] - prev_point[axis])
    intersection_point = alpha * prev_point + (1.0 - alpha) * current_point
    return intersection_point

  def _inside(self, plane, point, axis):
    x, y = axis
    u = plane[0] - point
    v = plane[1] - point

    a = u[x] * v[y]
    b = u[y] * v[x]
    return a >= b

  def _classify_point_to_plane(self, point, plane, normal, axis):
    signed_distance = normal * (point[axis] - plane[axis])
    if signed_distance > _PLANE_THICKNESS_EPSILON:
      return _POINT_IN_FRONT_OF_PLANE
    elif signed_distance < -_PLANE_THICKNESS_EPSILON:
      return _POINT_BEHIND_PLANE
    else:
      return _POINT_ON_PLANE

  @property
  def intersection_points(self):
    return self._intersection_points
