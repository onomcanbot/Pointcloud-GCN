import numpy as np
import open3d as o3d
from modules.box import box_decoding
from modules.nms import *

from modules.dataset import quaternion_from_euler

label_map = {'Background': 0, 'Car': 1, 'DontCare': 3}

color_map = np.array([(211,211,211), (255, 0, 0), (255,20,147), (65, 244, 101),
    (169, 244, 65), (65, 79, 244), (65, 181, 244), (229, 244, 66)],
    dtype=np.float32)
color_map = color_map/255.0

gt_color_map = {
    'Pedestrian': (0,255,255),
    'Person_sitting': (218,112,214),
    'Car': (154,205,50),
    'Truck':(255,215,0),
    #'Van': (255,20,147),
    'Van': (30,20,147),
    #'Tram': (250,128,114),
    'Tram': (123,30,114),
    'Misc': (128,0,128),
    'Cyclist': (255,165,0),
}

detect_color_map = {
    'Pedestrian': (1,0,0),
    'Person_sitting': (1,0,0),
    'Car': (1,0,0),
    'Truck': (1,0,0),
    'Van': (1,0,0),
    'Tram': (1,0,0),
    'Misc': (1,0,0),
    'Cyclist': (1,0,0),
}

def get_box(last_layer_points_xyz, pred_loc, box_probs, cls_labels, is_train=False, raw_pred=False):
    BOX_ENCODING_LEN = pred_loc.shape[-1]
    NUM_CLASSES = pred_loc.shape[-2]
    detection_boxes_3d = None
    box_indices_size =0 
    box_labels = []
    pred_boxes = []
    boxes_centers = []
    detection_scores = []
    print("box_probs 1",box_probs)
    if raw_pred:
        cls_labels = np.expand_dims(np.argmax(box_probs, axis=1), axis=1)

    if is_train or raw_pred:
        points_xyz = np.expand_dims(last_layer_points_xyz, axis=1)

        box_idx = np.arange(0, pred_loc.shape[0])
        box_idx = np.expand_dims(box_idx, axis=1)
        box_idx = np.int64(np.concatenate((box_idx, cls_labels), axis=1))

        for idx in range(cls_labels.shape[0]):
            cls_idx = int(cls_labels[idx])
            if cls_idx != 0 and cls_idx < pred_loc.shape[1] - 1:
                box_labels.append(cls_idx)
                pred_boxes.append([pred_loc[idx][cls_idx]])
                boxes_centers.append(points_xyz[idx][0])
                detection_scores.append(box_probs[idx][cls_idx])
       

        box_labels = np.asarray(box_labels)
        pred_boxes = np.asarray(pred_boxes)
        boxes_centers = np.asarray(boxes_centers)
        detection_scores = np.asarray(detection_scores)

        box_indices_size = box_labels.shape[0]

        if box_indices_size != 0:
            decoded_boxes = box_decoding(np.expand_dims(box_labels, axis=1),
            boxes_centers, pred_boxes, label_map)

            box_labels = np.squeeze(box_labels)
            decoded_boxes = np.squeeze(decoded_boxes)
    else:
        box_labels = np.tile(np.expand_dims(np.arange(NUM_CLASSES), axis=0),
            (box_probs.shape[0], 1))
        box_labels = box_labels.reshape((-1))
        #print("box_labels",box_labels)
        raw_box_labels = box_labels
        box_probs = box_probs.reshape((-1))
        #print("box_probs 2",box_probs)
        pred_boxes = pred_loc.reshape((-1, 1, BOX_ENCODING_LEN))
        #print("pred_boxes",pred_boxes[0])
        last_layer_points_xyz = np.tile(
        np.expand_dims(last_layer_points_xyz, axis=1), (1, NUM_CLASSES, 1))
        last_layer_points_xyz = last_layer_points_xyz.reshape((-1, 3))
        boxes_centers = last_layer_points_xyz
        decoded_boxes = box_decoding(np.expand_dims(box_labels, axis=1),
            boxes_centers, pred_boxes, label_map)

        #print("box_labels",box_labels.shape)
        #print("decoded_boxes",decoded_boxes)
        
        box_mask = (box_labels > 0)*(box_labels < NUM_CLASSES-1)
        box_mask = box_mask*(box_probs > 1./NUM_CLASSES)
        box_indices = np.nonzero(box_mask)[0]

        if box_indices.size != 0:
                box_labels = box_labels[box_indices]
                box_probs = box_probs[box_indices]
                #print("box_probs 3",box_probs)
                #print("box_labels",box_labels)
                #print("box_indices",box_indices)
                box_probs_ori = box_probs
                decoded_boxes = decoded_boxes[box_indices, 0]
                box_labels[box_labels==2]=1
                box_labels[box_labels==4]=3
                box_labels[box_labels==6]=5
                detection_scores = box_probs
                #print("detection_scores    1:",detection_scores)
                box_indices_size = len(box_indices)
                            
                #print(pred_boxes)
                
                #print("decoded_boxes",decoded_boxes)

    print("box_indices_size",box_indices_size)

    
    if box_indices_size > 1:
        (class_labels, detection_boxes_3d, detection_scores,
        nms_indices) = nms_boxes_3d_merge_only(
            box_labels, decoded_boxes, detection_scores,
            overlapped_fn=overlapped_boxes_3d_iou,
            overlapped_thres=0.01,
            appr_factor=100.0, top_k=-1,
            attributes=np.arange(box_indices_size)
            )
        print("detection_scores    2:",detection_scores)
            
        '''
        (class_labels, detection_boxes_3d, detection_scores,
        nms_indices) = nms_boxes_3d_uncertainty(
            box_labels, decoded_boxes, detection_scores,
            overlapped_fn=overlapped_boxes_3d_fast_poly,
            overlapped_thres=0.01,
            appr_factor=100.0, top_k=-1,
            attributes=np.arange(box_indices_size)
            )
        '''

    return detection_boxes_3d, detection_scores

class Visualizer:
    def __init__(self, dataset, init_geomery_list=[]):
        self.dataset = dataset
        self.vis = o3d.visualization.Visualizer()
        self.pcd = o3d.geometry.PointCloud()
        self.cls_pcd = o3d.geometry.PointCloud()
        self.gt_line_set = o3d.geometry.LineSet()
        self.dt_line_set = o3d.geometry.LineSet()
        self.gt_point = o3d.geometry.PointCloud()
        self.dt_point = o3d.geometry.PointCloud()

        self.initialize()

    def initialize(self):
        self.vis.create_window()
        self.pcd = o3d.io.read_point_cloud("./modules/sample/test_sample.ply")
        print("pcd",self.pcd)
        
        #self.pcd.points = o3d.utility.Vector3dVector()
        #self.pcd.colors = o3d.utility.Vector3dVector()

        self.cls_pcd.points = o3d.utility.Vector3dVector([[0, 0, 0]])
        self.cls_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1.0]])

        self.gt_line_set.points = o3d.utility.Vector3dVector([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]])
        self.gt_line_set.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
        self.gt_line_set.colors = o3d.utility.Vector3dVector([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

        self.dt_line_set.points = o3d.utility.Vector3dVector([[0, 0, 0]])
        self.dt_line_set.lines = o3d.utility.Vector2iVector([[0, 0]])

        self.gt_point.points = o3d.utility.Vector3dVector([[0, 0, 0]])
        self.gt_point.colors = o3d.utility.Vector3dVector([[1.0, 0, 0]])
        self.dt_point.points = o3d.utility.Vector3dVector([[0, 0, 0]])
        self.dt_point.colors = o3d.utility.Vector3dVector([[1.0, 0, 0]])

        self.vis.add_geometry(self.cls_pcd)
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.gt_line_set)
        self.vis.add_geometry(self.dt_line_set)
        self.vis.add_geometry(self.dt_point)
        self.vis.add_geometry(self.gt_point)

        ctr = self.vis.get_view_control()
        ctr.rotate(0.0, 3141.0, 0)

        self.vis.run()

    def box_point(self, point, box_points, gt=True):
        VPS_point_xyz = box_points         # ori X Y Z
        if gt:
            VPS_point_color = np.array([0, 255, 0])
        else:
            VPS_point_color = np.array([255,0,0])
        VPS_point_color = VPS_point_color/255
        

        point.points = o3d.utility.Vector3dVector(VPS_point_xyz)
        point.paint_uniform_color(VPS_point_color)

    def create_color_list(self, color, points_cnt):
        color_list = np.zeros((points_cnt, 3))
        color_list[:, 0] = color[0]
        color_list[:, 1] = color[1]
        color_list[:, 2] = color[2]
        print
        return color_list

    def set_base_pcd(self, points, colors, is_train=False):
        # base color off
        if is_train:
            base_color = [0.7, 0.7, 0.7]
            colors = self.create_color_list(base_color, points.shape[0])
        self.pcd.points = o3d.utility.Vector3dVector(points)
        #self.pcd.colors = o3d.utility.Vector3dVector(colors)


        #self.pcd.points = o3d.utility.Vector3dVector([[0, 0, 0]])
        #self.pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1.0]])

    def set_cls_point(self, cls_point):
        cls_color = [1.0, 0.0, 0.0]
        cls_color_list = self.create_color_list(cls_color, len(cls_point))

        self.cls_pcd.points = o3d.utility.Vector3dVector(cls_point)
        self.cls_pcd.colors = o3d.utility.Vector3dVector(cls_color_list)

    def set_gt_box(self, box_list,get_point=False):
        gt_box_point=self.draw_box(self.gt_line_set, box_list, is_gt=True)
        self.box_point(self.gt_point, gt_box_point, gt=True)
        if get_point:
            return gt_box_point

    def set_dt_box(self, box_list,get_point=False):
        dt_box_point=self.draw_box(self.dt_line_set, box_list)
        self.box_point(self.dt_point, dt_box_point, gt=False)
        if get_point:
            return dt_box_point

    def scene_draw(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.vis.add_geometry(self.cls_pcd)
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.gt_line_set)
        self.vis.add_geometry(self.dt_line_set)
        self.vis.add_geometry(self.gt_point)
        self.vis.add_geometry(self.dt_point)

        ctr = self.vis.get_view_control()
        ctr.rotate(0.0, 3141.0, 0)

        self.vis.run()

    def realtime_draw(self):
        # linux
        #self.vis.update_geometry()

        # Windows
        self.vis.update_geometry(self.pcd)
        self.vis.update_geometry(self.cls_pcd)
        self.vis.update_geometry(self.gt_line_set)
        self.vis.update_geometry(self.dt_line_set)
        self.vis.update_geometry(self.gt_point)
        self.vis.update_geometry(self.dt_point)
        self.vis.poll_events()
        self.vis.update_renderer()
    def get_gt_box_coners(self, box_list):
        return draw_box(self.gt_line_set, box_list, is_gt=True)
    def draw_box(self, line_set, box_list, is_gt=False):
        boxes_3d = []
        boxes_colors = []
        box_corners, box_edges, box_colors = None, None, None

        box_color_map = detect_color_map


        if is_gt:
            box_color_map = gt_color_map

            for label in box_list:
                if label['name'] in box_color_map:
                
                    pitch = label['pitch'] # pitch
                    yaw = label['yaw']     # yaw
                    roll = label['roll']   # roll
                
                    qw, qx, qy, qz = quaternion_from_euler(roll, pitch, yaw)
                    boxes_3d.append([
                        label['x3d'], label['y3d'], label['z3d'],
                        label['length'], label['height'], label['width'],
                        qw, qx, qy, qz])                    
                
                    boxes_colors.append(box_color_map[label['name']])
            
            boxes_3d = np.array(boxes_3d)
            boxes_colors = np.array(boxes_colors)/255.

            box_corners, box_edges, box_colors = \
                self.dataset.boxes_3d_to_line_set_qt(boxes_3d, boxes_color=boxes_colors)
        else:
            if box_list is not None:
                box_corners, box_edges, box_colors = \
                            self.dataset.boxes_3d_to_line_set_qt(box_list)


        box_points = []
        box_points.append(box_corners[0])
        box_points.append(box_corners[1])
        box_points.append(box_corners[2])
        box_points.append(box_corners[3])
        box_points.append(box_corners[4])
        box_points.append(box_corners[5])
        box_points.append(box_corners[6])
        box_points.append(box_corners[7])
        box_points.append(box_corners[8])

        if box_corners is None or box_corners.size<1:
            box_corners = np.array([[0, 0, 0]])
            box_edges = np.array([[0, 0]])
            box_colors =  np.array([[0, 0, 0]])

        line_set.points = o3d.utility.Vector3dVector(np.vstack(
            [box_corners]))
        line_set.lines = o3d.utility.Vector2iVector(np.vstack(
            [box_edges]))
        line_set.colors = o3d.utility.Vector3dVector(np.vstack(
            [box_colors]))
        return box_points
