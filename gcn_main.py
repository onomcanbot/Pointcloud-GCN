import os
import argparse
import math
import time

import numpy as np
from tqdm import trange
from tqdm import tqdm

from modules.dataset import KittiDataset, euler_from_quaternion
from modules.load_dataset import *
from modules.box import box_encoding, box_decoding
from modules.model_gcn import *
from modules.visualizer import *
from modules.nms import *
from util.config_util import load_train_config, load_config
from util.metrics import recall_precisions, mAP

import torch
import torch.nn as nn
import torch.optim as optim
#import tensorboardX import SummaryWriter

import open3d as o3d

if torch.cuda.is_available() == True:
    print("torch cuda is available")

TRAIN_CONFIG_PATH = "./configs/train_config"
CONFIG_PATH = "./configs/config"
DATASET_DIR = "./dataset/kitti"
DATASET_LIST = "../dataset/split.txt"

train_config = load_train_config(TRAIN_CONFIG_PATH)
config_complete = load_config(CONFIG_PATH)

if 'train' in config_complete:
    config = config_complete['train']
else:
    config = config_complete

#=============================================================
# GCN Training Config
#=============================================================
model_args = {
    "feature_size": 6,
    "hidden_size": 512,
    "n_block": 3,
    "cls_layer_list": [512, 256, 64],
    "loc_layer_list": [512, 256, 128, 64],
    "pred_cls_num": 3,
    "pred_loc_len": 10,
    "sc_type": "sc",
    "concat_feature": False,
    "gcn_act_fn": "relu",
    "bias": True,
    "use_batch_norm": True,
    "dropout_rate": 0,
    "use_rd": True,
    "rd_act_fn": "relu",
}

loss_args = {
    #"cls_loss_type": "focal_sigmoid",
    "cls_loss_type": "softmax",
    "loc_loss_type": "huber_loss",
    "cls_loss_weight": 0.1,
    "loc_loss_weight": 10.0
}

learning_rate = 0.008
decay_rate = 0.1
epoches = 2000
batch_size = 1
"""
device : None = CPU
device : cuda:$GPU_NUM = GPU
"""
device = None
#device = "cuda:0"

save_model_perid = 10

base_model_folder = "./model"
model_name = "Test"
current_epoch = 0

"""
TRAIN : 학습시 [True] / 시각화시 [False]
TRAIN_VIS : 학습시 실시간 시각화 [True]
RAW_PRED : cls, loc를 모두 예측값으로 시각화 [True] / cls는 정답, loc만 예측으로 시각화 [False]
"""
TRAIN = False
TRAIN_VIS = True
RAW_PRED = False

#=============================================================

# input function ==============================================================
dataset = KittiDataset(
    os.path.join(DATASET_DIR, 'image/training/image_2'),
    os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
    os.path.join(DATASET_DIR, 'calib/training/calib/'),
    os.path.join(DATASET_DIR, 'labels/training/label_2'),
    DATASET_LIST,
    num_classes=model_args["pred_cls_num"])

NUM_CLASSES = dataset.num_classes
BOX_ENCODING_LEN = model_args["pred_loc_len"]

#=============================================================

optimizer = None
model = GCN_Model(**model_args)

if device == "cuda:0":
    model = model.to(device)
NUM_TEST_SAMPLE = dataset.num_files

label_map = {'Background': 0, 'Car': 1, 'DontCare': 3}

def save_log(file_name, log_data, start=None, end=None):
    with open(file_name, mode="a", encoding="utf-8") as f:
        if start is not None and end is not None:
            additional = time.time()
            log_data += f", {end - start}, {additional - end}"
            log_data += "\n"
        f.write(log_data)

    return ""

def detect(frame_idx, is_train=False, raw_pred=False):
    batch_list = []

    fetch = fetch_data(dataset, frame_idx, train_config, config)

    batch_list.append(fetch)
    batch = batch_data(batch_list)

    new_batch = []

    if device == "cuda:0":
        for item in batch:
            if item == None:
                new_batch += [""]
                continue

            if not isinstance(item, torch.Tensor):
                item = [x.to(device) for x in item]
            else: 
                item = item.to(device)

            new_batch += [item]

        batch = new_batch

    input_v, vertex_coord_list, keypoint_indices_list, \
    edges_list, cls_labels, encoded_boxes, valid_boxes, \
    adj_local, adj_global, adj_global_relative, \
    feature_matrix_local, feature_matrix_global = batch

    last_layer_points_xyz = vertex_coord_list[-1]

    pred_cls, pred_loc = model(adj_global, adj_local, adj_global_relative, feature_matrix_local, feature_matrix_global)

    box_probs = model.get_prob(pred_cls)


    input_v, vertex_coord_list, keypoint_indices_list, \
    edges_list, cls_labels, encoded_boxes, valid_boxes, \
    adj_local, adj_global, adj_global_relative, \
    feature_matrix_local, feature_matrix_global = fetch

    last_layer_points_xyz = vertex_coord_list[-1]
    pred_cls = pred_cls.cpu().detach().numpy()
    pred_loc = pred_loc.cpu().detach().numpy()
    box_probs = box_probs.cpu().detach().numpy()
    
    detection_boxes_3d = get_box(last_layer_points_xyz, pred_loc, box_probs, cls_labels, is_train, raw_pred)

    pred_cls_idx = np.argmax(box_probs, axis=1)
    cls_mask = (pred_cls_idx > 0) * (pred_cls_idx < box_probs.shape[-1])
    cls_point = last_layer_points_xyz[cls_mask]

    print("=====================")
    print("# Detect")
    print("=====================")
    print(f"- Background : {np.where(pred_cls_idx == 0)[0].shape[0]}")
    
    for cls_idx in range(box_probs.shape[-1] - 2):
        print(f"- Class_{cls_idx+1} : {np.where(pred_cls_idx == cls_idx + 1)[0].shape[0]}")
    print(f"- DontCare : {np.where(pred_cls_idx == 3)[0].shape[0]}")
    print("=====================")
    print()

    """
    cls_point = []

    for idx in range(cls_mask.shape[0]):
        cls_idx = cls_mask[idx]
        if  cls_idx != 0 and cls_idx != box_probs.shape[-1] - 1:
            cls_point.append(last_layer_points_xyz[idx])

    cls_point = np.asarray(cls_point)
    """

    return input_v, vertex_coord_list, detection_boxes_3d, cls_point


def train(save_path, VIS=None, train_vis=False, raw_pred=False, current_epoch=0, decay_rate=None):
    os.makedirs(save_path, exist_ok=True)

    #surmmary = SummaryWriter()
    total_step = current_epoch * (NUM_TEST_SAMPLE - batch_size + 1)
    train_log1 = ""
    train_log2 = ""
    lr_decay = False

    for epoch in range(1 + current_epoch, epoches):
        recalls_list, precisions_list, mAP_list, cls_idx_loc_loss_list = {}, {}, {}, {}

        for i in range(NUM_CLASSES):
            recalls_list[i], precisions_list[i], mAP_list[i], cls_idx_loc_loss_list[i] = [], [], [], []

        frame_idx_list = np.random.permutation(NUM_TEST_SAMPLE)

        #pbar = tqdm(list(range(0, NUM_TEST_SAMPLE-batch_size+1, batch_size)), desc="start training", leave=True)
        pbar = tqdm(list(range(0, NUM_TEST_SAMPLE-batch_size+1, batch_size)))
        print("=================================================================================")
        batch_start = time.time()
        for batch_idx in pbar:

            frame_idx = frame_idx_list[batch_idx]

            batch_list = []
            fetch = fetch_data(dataset, frame_idx, train_config, config)
            batch_list.append(fetch)
            batch = batch_data(batch_list)

            new_batch = []

            if device == "cuda:0":
                for item in batch:
                    if item == None:
                        new_batch += [""]
                        continue

                    if not isinstance(item, torch.Tensor):
                        item = [x.to(device) for x in item]
                    else: 
                        item = item.to(device)

                    new_batch += [item]

                batch = new_batch

            input_v, vertex_coord_list, keypoint_indices_list, \
            edges_list, cls_labels, encoded_boxes, valid_boxes, \
            adj_local, adj_global, adj_global_relative, \
            feature_matrix_local, feature_matrix_global = batch

            pred_cls, pred_loc = model(adj_global, adj_local, adj_global_relative, feature_matrix_local, feature_matrix_global)
            predictions = torch.argmax(pred_cls, dim=1)

            loss_dict = model.loss(pred_cls, cls_labels, pred_loc, encoded_boxes, valid_boxes, **loss_args)
            t_cls_loss, t_loc_loss, t_reg_loss = loss_dict['cls_loss'], loss_dict['loc_loss'], loss_dict['reg_loss']

            cls_loss = "{0:01.3f}".format(t_cls_loss)
            loc_loss = "{0:01.3f}".format(t_loc_loss)
            reg_loss = "{0:01.3f}".format(t_reg_loss)
            pbar.set_description(f"# Epoch (Step) : {epoch} ({batch_idx + 1}), t_cls_loss : {cls_loss}, t_loc_loss : {loc_loss}, t_reg_loss : {reg_loss}")

            total_step += 1
            train_log1 += f"{total_step}, {cls_loss}, {loc_loss}, {reg_loss}\n"

            t_total_loss = t_cls_loss + t_loc_loss + t_reg_loss
            optimizer.zero_grad()
            t_total_loss.backward()
            optimizer.step()

            cls_probs = None

            if loss_args["cls_loss_type"] == "softmax":
                cls_probs = model.get_prob(pred_cls)
            else:
                cls_probs = pred_cls.sigmoid()

            # record metrics
            recalls, precisions = recall_precisions(cls_labels, predictions, NUM_CLASSES)
            mAPs = mAP(cls_labels, cls_probs, NUM_CLASSES)



            print("mAPs", mAPs)

            # mAPs {0: 0.9861552984270133, 1: 0.055769037947148034, 2: 0.0}

            
            
            cls_idx_loc_loss = loss_dict["classwise_loc_loss"]

            for i in range(NUM_CLASSES):
                recalls_list[i] += [recalls[i]]
                precisions_list[i] += [precisions[i]]
                mAP_list[i] += [mAPs[i]]
                idx_loc_loss = cls_idx_loc_loss[i].cpu().detach().numpy()
                cls_idx_loc_loss_list[i] += [np.sum(idx_loc_loss)]


            print("mAP_list", mAP_list)

            """
            mAP_list {0: [0.9861552984270133, 0.9536609005299203], 
                      1: [0.055769037947148034, 0.01598924109500763], 
                      2: [0.0, 0.0]}
            """
             
             
        print("=================================================================================")
        batch_end = time.time()

        # print metrics
        for class_idx in range(NUM_CLASSES):
            m_recall = "{0:01.3f}".format(np.mean(recalls_list[class_idx]))
            m_precision = "{0:01.3f}".format(np.mean(precisions_list[class_idx]))
            m_mAP = "{0:01.3f}".format(np.mean(mAP_list[class_idx]))
            m_cls_idx_loc_loss = "{0:01.3f}".format(np.mean(cls_idx_loc_loss_list[class_idx]))

            print(f"- class_idx:{class_idx} > recall: {m_recall}, precision: {m_precision}, mAP: {m_mAP}, loc_loss: {m_cls_idx_loc_loss}")

            if class_idx != 0 and float(m_recall) > 0.88 and float(m_precision) > 0.88:
                lr_decay = True

            train_log2 +=  f"{epoch}, {m_recall}, {m_precision}, {m_mAP}, {m_cls_idx_loc_loss}"

        print()

        train_log2 += f", {optimizer.param_groups[0]['lr']}"

        train_log1 = save_log(f"{save_path}/train_log1.txt", train_log1)
        train_log2 = save_log(f"{save_path}/train_log2.txt", train_log2, batch_start, batch_end)

        if epoch % save_model_perid == 0 or lr_decay:
            #torch.save(model, "saved_models/model_{}.pt".format(epoch))
            torch.save(model.state_dict(), f"{save_path}/model_state_{epoch}.pt")

        if VIS != None and train_vis:
            input_v, vertex_coord_list, detection_boxes_3d, cls_point = detect(frame_idx_list[0], is_train=True, raw_pred=raw_pred)

            gt_box_label_list = dataset.get_ply_label(frame_idx_list[0])

            VIS.set_cls_point(cls_point)
            VIS.set_base_pcd(vertex_coord_list[0], input_v[:, 1:], is_train=True)
            VIS.set_gt_box(gt_box_label_list)
            VIS.set_dt_box(detection_boxes_3d)
            
            VIS.realtime_draw()

        if decay_rate is not None and lr_decay:
            lr_decay = False
            optimizer.param_groups[0]['lr'] *= decay_rate

if __name__ == "__main__":

    frame_idx_list = np.random.permutation(NUM_TEST_SAMPLE)
    VIS = None

    pre_model = ""
    if current_epoch != 0:
        pre_model = os.path.join(base_model_folder, model_name, (f"model_state_{current_epoch}.pt"))

    if TRAIN:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if pre_model != "":
            model.load_state_dict(torch.load(pre_model))

        if TRAIN_VIS:
            VIS = Visualizer(dataset)

        save_path = os.path.join(base_model_folder, model_name)
        train(save_path, VIS, TRAIN_VIS, RAW_PRED, current_epoch, decay_rate)

        
    else:
        VIS = Visualizer(dataset)
        model.load_state_dict(torch.load(pre_model))
        #model.eval()

        for frame_idx in frame_idx_list:
            input_v, vertex_coord_list, detection_boxes_3d, cls_point = detect(frame_idx)

            gt_box_label_list = dataset.get_ply_label(frame_idx)

            VIS.set_cls_point(cls_point)
            VIS.set_base_pcd(vertex_coord_list[0], input_v[:, 1:])
            VIS.set_gt_box(gt_box_label_list)
            VIS.set_dt_box(detection_boxes_3d)

            #VIS.realtime_draw()
            VIS.scene_draw()

    
