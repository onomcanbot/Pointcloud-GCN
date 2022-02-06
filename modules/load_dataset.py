import numpy as np
import scipy.sparse as sp

import torch
from modules import preprocess
from modules.box import box_encoding, box_decoding, classaware_all_class_box_encoding
from modules.dataset import KittiDataset
from modules.graph_gen import get_graph_generate_fn
from multiprocessing import Pool, Queue, Process
import os
import argparse
from util.config_util import save_config, save_train_config, \
    load_train_config, load_config

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def convert_sparse_matrix_v1(vertex_coord_list, keypoint_indices_list, edges_list):
    key_idx = np.squeeze(keypoint_indices_list[0])

    edges_list[0][:, 1] = np.take(key_idx, edges_list[0][:, 1])

    #edges_list[1][:, 0] = np.take(key_idx, edges_list[1][:, 0])
    #edges_list[1][:, 1] = np.take(key_idx, edges_list[1][:, 1])

    keypoint_idx = keypoint_indices_list[1]
    loop = np.array([keypoint_idx, keypoint_idx]).transpose().squeeze()
    edges_list[1] = np.concatenate((edges_list[1], loop), axis=0)

    adj_local = sp.coo_matrix((np.ones(edges_list[0].shape[0]), (edges_list[0][:, 0], edges_list[0][:, 1])),
                        shape=(vertex_coord_list[0].shape[0], vertex_coord_list[0].shape[0]),
                        dtype=np.float32)

    adj_global = sp.coo_matrix((np.ones(edges_list[1].shape[0]), (edges_list[1][:, 0], edges_list[1][:, 1])),
                        shape=(vertex_coord_list[1].shape[0], vertex_coord_list[1].shape[0]),
                        dtype=np.float32)

    #print(adj_global)

    return adj_local, adj_global

def convert_sparse_matrix_v2(vertex_coord_list, keypoint_indices_list, edges_list):
    ori_key_idx = np.squeeze(keypoint_indices_list[0])
    new_key_idx = np.squeeze(keypoint_indices_list[1])
    edge_local = edges_list[0]
    edge_global = edges_list[1]
    full_vertex = vertex_coord_list[0]
    key_vertex = vertex_coord_list[-1]

    # Local_Graph : Keypoint Index -> Original Vertex Index
    edge_local[:, 1] = ori_key_idx[edge_local[:, 1]]

    # Add self attention to graph
    local_loop = np.array([[ori_key_idx], [ori_key_idx]]).transpose().squeeze()
    edge_local = np.concatenate((edge_local, local_loop), axis=0)

    global_loop = np.array([[new_key_idx], [new_key_idx]]).transpose().squeeze()
    edge_global = np.concatenate((edge_global, global_loop), axis=0)

    # Get graph norm
    local_near = full_vertex[edge_local[:, 0]]
    local_key = full_vertex[edge_local[:, 1]]
    local_norm = np.linalg.norm((local_near - local_key), axis=1, ord=2)
    local_max_norm = np.amax(local_norm)

    global_near = key_vertex[edge_global[:, 0]]
    global_key = key_vertex[edge_global[:, 1]]
    global_norm = np.linalg.norm((global_near - global_key), axis=1, ord=2)
    global_max_norm = np.amax(global_norm)

    adj_local = sp.coo_matrix((np.ones(edge_local.shape[0]), (edge_local[:, 0], edge_local[:, 1])),
                        shape=(full_vertex.shape[0], full_vertex.shape[0]),
                        dtype=np.float32)

    adj_global = sp.coo_matrix((np.ones(edge_global.shape[0]), (edge_global[:, 0], edge_global[:, 1])),
                        shape=(key_vertex.shape[0], key_vertex.shape[0]),
                        dtype=np.float32)

    return adj_local, adj_global

def get_adj_matrix_v1(vertex, keypoint, edge, is_local=False, norm_type=2):
    key_idx = np.squeeze(keypoint)

    #print(edge)

    if is_local:
        # Local_Graph : Keypoint Index -> Original Vertex Index
        edge[:, 1] = key_idx[edge[:, 1]]

    # Add self attention to graph
    loop = np.array([[key_idx], [key_idx]]).transpose().squeeze()
    edge = np.concatenate((edge, loop), axis=0)

    # Get graph norm
    near_point = vertex[edge[:, 0]]
    key_point = vertex[edge[:, 1]]
    norm = np.linalg.norm((near_point - key_point), axis=1, ord=norm_type)
    max_norm = np.amax(norm)

    # Normalization (y = -x + 1)
    norm = -(norm / max_norm) + 1

    adj_matrix = sp.coo_matrix((norm, (edge[:, 0], edge[:, 1])),
                        shape=(vertex.shape[0], vertex.shape[0]),
                        dtype=np.float32)

    return adj_matrix

def get_feature_matrix_v1(point_features, point_coordinates, keypoint_indices, edges_list):
    keypoint_coord = point_coordinates[1]

    keypoint_feature = point_features[keypoint_indices[0]]
    keypoint_feature = np.reshape(keypoint_feature, [keypoint_feature.shape[0], keypoint_feature.shape[2]])

    #print(keypoint_coord.shape)
    #print(keypoint_feature.shape)

    feature_matrix = np.concatenate((keypoint_coord, keypoint_feature), axis=1)    

    #print(feature_matrix.shape)

    return feature_matrix

def get_adj_matrix(vertex, keypoint, edge, edge_num, is_local=False, norm_type=2):
    key_idx = np.squeeze(keypoint)

    #print(edge)

    if is_local:
        # Local_Graph : Keypoint Index -> Original Vertex Index
        edge[:, 1] = key_idx[edge[:, 1]]

    # Add self attention to graph
    loop = np.array([[key_idx], [key_idx]]).transpose().squeeze()
    loop_data = -1 * np.array(edge_num)

    #print("loop")
    #print(loop)

    # Create edge connection
    edge_data = np.ones(edge.shape[0])

    adj_matrix_normal = sp.coo_matrix((edge_data, (edge[:, 0], edge[:, 1])),
                            shape=(vertex.shape[0], vertex.shape[0]),
                            dtype=np.float32)

    # Concat edge and data
    edge = np.concatenate((edge, loop), axis=0)
    edge_data = np.concatenate((edge_data, loop_data), axis=0)

    adj_matrix_relative = sp.coo_matrix((edge_data, (edge[:, 0], edge[:, 1])),
                        shape=(vertex.shape[0], vertex.shape[0]),
                        dtype=np.float32)

    return adj_matrix_normal, adj_matrix_relative

def get_feature_matrix(point_features, point_coordinates, keypoint_indices, edges_list):
    keypoint_coord = point_coordinates[1]

    keypoint_feature = point_features[keypoint_indices[0]]
    keypoint_feature = np.reshape(keypoint_feature, [keypoint_feature.shape[0], keypoint_feature.shape[2]])

    #print(keypoint_coord.shape)
    #print(keypoint_feature.shape)

    global_feature_matrix = np.concatenate((keypoint_coord, keypoint_feature), axis=1)    

    #print(feature_matrix.shape)

    local_coord = point_coordinates[0]
    local_feature_matrix = np.concatenate((local_coord, point_features), axis=1)

    return local_feature_matrix, global_feature_matrix


def fetch_data(dataset, frame_idx, train_config, config):
    
    aug_fn = preprocess.get_data_aug(train_config['data_aug_configs'])
    
    BOX_ENCODING_LEN = 10
    
    graph_generate_fn= get_graph_generate_fn(config['graph_gen_method'])

    cam_rgb_points = dataset.get_ply_point(frame_idx)
    box_label_list = dataset.get_ply_label(frame_idx)
    
    (vertex_coord_list, keypoint_indices_list, edges_list, edges_num_list) = \
        graph_generate_fn(cam_rgb_points.xyz, **config['graph_gen_kwargs'])

    local_idx = 0
    global_idx = -1

    adj_local, adj_local_relative = get_adj_matrix(vertex_coord_list[local_idx], keypoint_indices_list[local_idx], edges_list[local_idx], edges_num_list[local_idx], is_local=True, norm_type=2)
    adj_global, adj_global_relative = get_adj_matrix(vertex_coord_list[global_idx], keypoint_indices_list[global_idx],
                                                                                    edges_list[global_idx], edges_num_list[global_idx],  is_local=False, norm_type=2)

    config['input_features'] = "rgb"

    if config['input_features'] == 'rgb':
        input_v = cam_rgb_points.attr

    elif config['input_features'] == '0':
        input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))

    feature_matrix_local, feature_matrix_global = get_feature_matrix(input_v, vertex_coord_list, keypoint_indices_list, edges_list)

    last_layer_points_xyz = vertex_coord_list[-1]
    
    if config['label_method'] == 'Car':

        cls_labels, boxes_3d, valid_boxes, label_map = \
            dataset.assign_classaware_car_label_to_points(box_label_list,
            last_layer_points_xyz,
            expend_factor=train_config.get('expend_factor', (1.0, 1.0, 1.0)))

    encoded_boxes = box_encoding(cls_labels, last_layer_points_xyz,
        boxes_3d, label_map)

    input_v = input_v.astype(np.float32)
    vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
    keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
    edges_list = [e.astype(np.int32) for e in edges_list]

    cls_labels = cls_labels.astype(np.float32)
    encoded_boxes = encoded_boxes.astype(np.float32)
    valid_boxes = valid_boxes.astype(np.float32)
    return(input_v, vertex_coord_list, keypoint_indices_list, edges_list,
        cls_labels, encoded_boxes, valid_boxes, adj_local_relative, adj_global, 
        adj_global_relative, feature_matrix_local, feature_matrix_global)

def batch_data(batch_list):
    input_v, vertex_coord_list, keypoint_indices_list, \
    edges_list, cls_labels, encoded_boxes, valid_boxes, \
    adj_local, adj_global, adj_global_relative, \
    feature_matrix_local, feature_matrix_global = batch_list[0]

    batched_input_v = torch.from_numpy(input_v)
    batched_vertex_coord_list = [torch.from_numpy(item) for item in vertex_coord_list]
    batched_keypoint_indices_list = [torch.from_numpy(item).long() for item in keypoint_indices_list]
    batched_edges_list = [torch.from_numpy(item).long() for item in edges_list]
    batched_cls_labels = torch.from_numpy(cls_labels)
    batched_encoded_boxes = torch.from_numpy(encoded_boxes)
    batched_valid_boxes = torch.from_numpy(valid_boxes)

    batched_adj_local = sparse_mx_to_torch_sparse_tensor(adj_local)
    batched_adj_global = sparse_mx_to_torch_sparse_tensor(adj_global)
    batched_adj_global_relative = sparse_mx_to_torch_sparse_tensor(adj_global_relative)
    batched_feature_matrix_local = torch.FloatTensor(feature_matrix_local)
    batched_feature_matrix_global = torch.FloatTensor(feature_matrix_global)

    return (batched_input_v, batched_vertex_coord_list,
        batched_keypoint_indices_list, batched_edges_list, batched_cls_labels,
        batched_encoded_boxes, batched_valid_boxes,
        batched_adj_local, batched_adj_global, batched_adj_global_relative,
        batched_feature_matrix_local, batched_feature_matrix_global)

def batch_data_old(batch_list):
    N_input_v, N_vertex_coord_list, N_keypoint_indices_list, N_edges_list,\
    N_cls_labels, N_encoded_boxes, N_valid_boxes, \
    N_adj_local, N_adj_global, N_feature_matrix = zip(*batch_list)

    batch_size = len(batch_list)
    level_num = len(N_vertex_coord_list[0])
    batched_keypoint_indices_list = []
    batched_edges_list = []

    for level_idx in range(level_num-1):
        centers = []
        vertices = []
        point_counter = 0
        center_counter = 0
        for batch_idx in range(batch_size):
            centers.append(
                N_keypoint_indices_list[batch_idx][level_idx]+point_counter)
            vertices.append(np.hstack(
                [N_edges_list[batch_idx][level_idx][:,[0]]+point_counter,
                 N_edges_list[batch_idx][level_idx][:,[1]]+center_counter]))
            point_counter += N_vertex_coord_list[batch_idx][level_idx].shape[0]
            center_counter += \
                N_keypoint_indices_list[batch_idx][level_idx].shape[0]
        batched_keypoint_indices_list.append(np.vstack(centers))
        batched_edges_list.append(np.vstack(vertices))
    batched_vertex_coord_list = []
    for level_idx in range(level_num):
        points = []
        counter = 0
        for batch_idx in range(batch_size):
            points.append(N_vertex_coord_list[batch_idx][level_idx])
        batched_vertex_coord_list.append(np.vstack(points))

    batched_input_v = np.vstack(N_input_v)
    batched_cls_labels = np.vstack(N_cls_labels)
    batched_encoded_boxes = np.vstack(N_encoded_boxes)
    batched_valid_boxes = np.vstack(N_valid_boxes)

    batched_adj_local = np.vstack(N_adj_local)
    batched_adj_global = np.vstack(N_adj_global)
    batched_feature_matrix = np.vstack(N_feature_matrix)

    batched_input_v = torch.from_numpy(batched_input_v)
    batched_vertex_coord_list = [torch.from_numpy(item) for item in batched_vertex_coord_list]
    batched_keypoint_indices_list = [torch.from_numpy(item).long() for item in batched_keypoint_indices_list]
    batched_edges_list = [torch.from_numpy(item).long() for item in batched_edges_list]
    batched_cls_labels = torch.from_numpy(batched_cls_labels)
    batched_encoded_boxes = torch.from_numpy(batched_encoded_boxes)
    batched_valid_boxes = torch.from_numpy(batched_valid_boxes)

    batched_adj_local = [sparse_mx_to_torch_sparse_tensor(item) for item in batched_adj_local]
    batched_adj_global = [sparse_mx_to_torch_sparse_tensor(item) for item in batched_adj_global]
    batched_feature_matrix = [torch.FloatTensor(item) for item in batched_feature_matrix]

    return (batched_input_v, batched_vertex_coord_list,
        batched_keypoint_indices_list, batched_edges_list, batched_cls_labels,
        batched_encoded_boxes, batched_valid_boxes,
        batched_adj_local, batched_adj_global, batched_feature_matrix)

class DataProvider(object):

    def __init__(self, dataset, train_config, config, async_load_rate=1.0, result_pool_limit=10000):
        if 'NUM_TEST_SAMPLE' not in train_config:
            self.NUM_TEST_SAMPLE = dataset.num_files
        else:
            if train_config['NUM_TEST_SAMPLE'] < 0:
                self.NUM_TEST_SAMPLE = dataset.num_files
            else:
                self.NUM_TEST_SAMPLE = train_config['NUM_TEST_SAMPLE']
        load_dataset_to_mem=train_config['load_dataset_to_mem']
        load_dataset_every_N_time=train_config['load_dataset_every_N_time']
        capacity=train_config['capacity']
        num_workers=train_config['num_load_dataset_workers']
        preload_list=list(range(self.NUM_TEST_SAMPLE))

        self.dataset = dataset
        self.train_config = train_config
        self.config = config
        self._fetch_data = fetch_data
        self._batch_data = batch_data
        self._buffer = {}
        self._results = {}
        self._load_dataset_to_mem = load_dataset_to_mem
        self._load_every_N_time = load_dataset_every_N_time
        self._capacity = capacity
        self._worker_pool = Pool(processes=num_workers)
        self._preload_list = preload_list
        self._async_load_rate = async_load_rate
        self._result_pool_limit = result_pool_limit

    def preload(self, frame_idx_list):
        """async load dataset into memory."""
        for frame_idx in frame_idx_list:
            result = self._worker_pool.apply_async(
                self._fetch_data, (self.dataset, frame_idx, self.train_config, self.config))
            self._results[frame_idx] = result

    def async_load(self, frame_idx):
        """async load a data into memory"""
        if frame_idx in self._results:
            data = self._results[frame_idx].get()
            del self._results[frame_idx]
        else:
            data = self._fetch_data(self.dataset, frame_idx, self.train_config, self.config)
        if np.random.random() < self._async_load_rate:
            if len(self._results) < self._result_pool_limit:
                result = self._worker_pool.apply_async(
                    self._fetch_data, (self.dataset, frame_idx, self.train_config, self.config))
                self._results[frame_idx] = result
        return data

    def provide(self, frame_idx):
        if self._load_dataset_to_mem:
            if self._load_every_N_time >= 1:
                extend_frame_idx = frame_idx+np.random.choice(
                    self._capacity)*self.NUM_TEST_SAMPLE
                if extend_frame_idx not in self._buffer:
                    data = self.async_load(frame_idx)
                    self._buffer[extend_frame_idx] = (data, 0)
                data, ctr = self._buffer[extend_frame_idx]
                if ctr == self._load_every_N_time:
                    data = self.async_load(frame_idx)
                    self._buffer[extend_frame_idx] = (data, 0)
                data, ctr = self._buffer[extend_frame_idx]
                self._buffer[extend_frame_idx] = (data, ctr+1)
                return data
            else:
                # do not buffer
                return self.async_load(frame_idx)
        else:
            return self._fetch_data(self.dataset, frame_idx, self.train_config, self.config)

    def provide_batch(self, frame_idx_list):
        batch_list = []
        for frame_idx in frame_idx_list:
            batch_list.append(self.provide(frame_idx))
        return self._batch_data(batch_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of PointGNN')
    parser.add_argument('train_config_path', type=str,
                       help='Path to train_config')
    parser.add_argument('config_path', type=str,
                       help='Path to config')
    parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/',
                       help='Path to KITTI dataset. Default="../dataset/kitti/"')
    parser.add_argument('--dataset_split_file', type=str,
                        default='',
                       help='Path to KITTI dataset split file.'
                       'Default="DATASET_ROOT_DIR/3DOP_splits'
                       '/train_config["train_dataset"]"')
    
    args = parser.parse_args()
    train_config = load_train_config(args.train_config_path)
    DATASET_DIR = args.dataset_root_dir
    config_complete = load_config(args.config_path)
    if 'train' in config_complete:
        config = config_complete['train']
    else:
        config = config_complete

    if args.dataset_split_file == '':
        DATASET_SPLIT_FILE = os.path.join(DATASET_DIR,
            './3DOP_splits/'+train_config['train_dataset'])
    else:
        DATASET_SPLIT_FILE = args.dataset_split_file

    # input function ==============================================================
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/training/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/training/calib/'),
        os.path.join(DATASET_DIR, 'labels/training/label_2'),
        DATASET_SPLIT_FILE,
        num_classes=config['num_classes'])

    data_provider = DataProvider(dataset, train_config, config)

    input_v, vertex_coord_list, keypoint_indices_list, edges_list, \
            cls_labels, encoded_boxes, valid_boxes = data_provider.provide_batch([1545, 1546])

    print(f"input_v: {input_v.shape}")
    for i, vertex_coord in enumerate(vertex_coord_list):
        print(f"vertex_coord: {i}: {vertex_coord.shape}")

    for i, indices in enumerate(keypoint_indices_list):
        print(f"indices: {i}: {indices.shape}")
        print(indices)
    for i, edge in enumerate(edges_list):
        print(f"edge: {i}: {edge.shape}")
        print(edge)
        #for item in edge:
        #    if item[0]==item[1]: print(item)
    print(f"cls_labels:{cls_labels.shape}")
    print(f"encoded_boxes: {encoded_boxes.shape}")
    print(f"valid_boxes: {valid_boxes.shape}")
    print(valid_boxes)
    print(f"max: {valid_boxes.max()}, min:{valid_boxes.min()}, sum: {valid_boxes.sum()}")

