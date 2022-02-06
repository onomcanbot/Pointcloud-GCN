import math
from collections import deque

import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
#from torch_scatter import scatter_max

def get_act_fn(act_fn):
    act_fn_list = {
        "relu": F.relu,
        "relu6": F.relu6,
        "leaky_relu": F.leaky_relu,
        "tanh": F.tanh
    }

    if act_fn == None:
        return None

    return act_fn_list[act_fn]

def get_loss_type(loss_type):
    loss_list = {
        "focal_sigmoid": focal_sigmoid,
        "softmax": sparse_softmax_cross_entropy_with_logits_pytorch,
        "huber_loss": nn.SmoothL1Loss(reduction="none"),
    }

    return loss_list[loss_type]

def focal_sigmoid(logits, labels, alpha=0.5, gamma=2):
    prob = logits.sigmoid()

    labels = F.one_hot(labels.squeeze().long(), num_classes=prob.shape[1])

    cross_entropy = torch.clamp(logits, min=0) - logits * labels + torch.log(1 + torch.exp(-torch.abs(logits)))

    prob_t = (labels * prob) + (1 - labels) * (1 - prob)
    modulating = torch.pow(1 - prob_t, gamma)
    alpha_weight = (labels * alpha) + (1 - labels) * (1 - alpha)

    focal_loss_cross_entropy = modulating * alpha_weight * cross_entropy

    return focal_loss_cross_entropy

def sparse_softmax_cross_entropy_with_logits_pytorch(logits, labels):
    """
    # onehot
    labels = labels.squeeze().long()
    num_classes = logits.shape[1]
    labels_onehot = torch.zeros(labels.shape[0], num_classes, device=labels.device).scatter_(1, labels.view(-1, 1), 1)
    """

    num_classes = logits.shape[1]
    labels = F.one_hot(labels.squeeze().long(), num_classes=num_classes)

    softmax_loss_cross_entropy = torch.sum(-labels * F.log_softmax(logits, -1), -1)
    #mean_loss = softmax_loss_cross_entropy.mean()

    return softmax_loss_cross_entropy

def fc_mlp_layer(layer_list, output_dim, is_logits=False):
    linears = []
    for L in range(len(layer_list) - 1):
        linears += [
            nn.Linear(layer_list[L], layer_list[L + 1]),
            nn.ReLU(),
            nn.BatchNorm1d(layer_list[L + 1])
        ]

    if is_logits:
        linears += [
            nn.Linear(layer_list[-1], output_dim)
        ]
    else:
        linears += [
            nn.Linear(layer_list[-1], output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim)
        ]

    return nn.Sequential(*linears)


def sparse_matrix_multiplication(adj, feature):
    new_feature = torch.spmm(adj, feature)

    for i in range(new_feature.shape[0]):
        new_feature[i] = new_feature[i]

    return new_feature


class GraphConvolutionLayer(Module):
    def __init__(self, feature_size, kernel_size, bias=True):
        super(GraphConvolutionLayer, self).__init__()

        self.feature_size = feature_size
        self.kernel_size = kernel_size
        self.weight = Parameter(torch.FloatTensor(feature_size, kernel_size))
        if bias:
            self.bias =  Parameter(torch.FloatTensor(kernel_size))
        else:
            self.register_parameter("bias", None)
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, feature):
        featureNweight = torch.mm(feature, self.weight)
        new_feature = torch.spmm(adj, featureNweight)

        if self.bias is not None:
            return new_feature + self.bias
        else:
            return new_feature

class SkipConnection(Module):
    def __init__(self, pre_feature_dim, new_feature_dim, concat_feature=False, use_gate=False):
        super(SkipConnection, self).__init__()

        self.input_dim = pre_feature_dim
        self.output_dim = new_feature_dim
        self.concat_feature = concat_feature
        self.use_gate = use_gate

        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.gate_linear_in = None
        self.gate_linear_out =None
        self.gate_fn = None

        if self.use_gate:
            self.gate_linear_in = nn.Linear(self.output_dim, self.output_dim)
            self.gate_linear_out = nn.Linear(self.output_dim, self.output_dim)
            self.gate_fn = nn.Sigmoid()


    def forward(self, pre_feature, new_feature):
        if self.input_dim != self.output_dim:
            pre_feature = self.linear(pre_feature)

        if self.use_gate:
            gate_value = self.pass_gate(pre_feature, new_feature)

            if self.concat_feature:
                new_feature = torch.cat((torch.mul(gate_value, new_feature), torch.mul(1.0 - gate_value, pre_feature)), dim=1)
            else:
                new_feature = torch.mul(gate_value, new_feature) + torch.mul(1.0 - gate_value, pre_feature)
        else:
            if self.concat_feature:
                new_feature = torch.cat((pre_feature, new_feature), dim=1)
            else:
                new_feature = pre_feature + new_feature

        return new_feature

    def pass_gate(self, pre_feature, new_feature):
        x1 = self.gate_linear_in(pre_feature)
        x2 = self.gate_linear_out(new_feature)
        return self.gate_fn(x1 + x2)

class ReadOut(Module):
    def __init__(self, input_dim, output_dim, act_fn=None, use_bn=False):
        super(ReadOut, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bn = use_bn

        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.batch_norm = nn.BatchNorm1d(self.output_dim)

        self.act_fn = get_act_fn(act_fn)

    def forward(self, feature):
        new_feature = self.linear(feature)

        #new_feature = torch.sum(new_feature, 1)

        if self.act_fn != None:
            new_feature = self.act_fn(new_feature)

        if self.use_bn:
            new_feature = self.batch_norm(new_feature)

        return new_feature

class Predictor(Module):
    def __init__(self, pred_cls_num, pred_loc_len, 
                            cls_layer_list=[300, 64], loc_layer_list=[300, 300, 64]):
        super(Predictor, self).__init__()

        self.pred_cls_num = pred_cls_num
        self.pred_loc_len = pred_loc_len

        self.pred_cls_fn = fc_mlp_layer(cls_layer_list, self.pred_cls_num, True)
        self.pred_loc_fns = nn.ModuleList()

        for _ in range(self.pred_cls_num):
            self.pred_loc_fns +=[
                fc_mlp_layer(loc_layer_list, self.pred_loc_len, True)
            ]

    def forward(self, feature):

        pred_cls = self.pred_cls_fn(feature)
        pred_loc_list = []

        for pred_loc_fn in self.pred_loc_fns:
            pred_loc = pred_loc_fn(feature).unsqueeze(1)
            pred_loc_list += [pred_loc]

        pred_loc = torch.cat(pred_loc_list, dim=1)

        return pred_cls, pred_loc

class GraphConvolutionBlock(Module):
    def __init__(self, feature_size, hidden_size,
                            sc_type=None, act_fn="relu", bias=True,
                            use_batch_norm=False, dropout_rate=0,
                            concat_feature=False):
        super(GraphConvolutionBlock, self).__init__()

        self.gcn_layer = GraphConvolutionLayer(feature_size, hidden_size, bias)

        self.sc_type = sc_type
        self.act_fn = get_act_fn(act_fn)
        self.use_batch_norm = use_batch_norm
        self.batch_norm = None
        self.dropout_rate = dropout_rate
        self.dropout = None

        if self.sc_type == "sc":
            self.sc_type = SkipConnection(feature_size, hidden_size, concat_feature)
        elif self.sc_type == "gated_sc":
            self.sc_type = SkipConnection(feature_size, hidden_size, concat_feature, use_gate=True)

        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate)

    def forward(self, adj, feature):
        pre_feature = feature

        new_feature = self.gcn_layer(adj, feature)

        if self.use_batch_norm:
            new_feature = self.batch_norm(new_feature)

        if self.act_fn != None:
            new_feature = self.act_fn(new_feature)

        if self.dropout_rate > 0:
            new_feature = self.dropout(new_feature)

        if self.sc_type != None:
            new_feature = self.sc_type(pre_feature, new_feature)

        return new_feature

class GCN_Model(Module):
    def __init__(self, feature_size, hidden_size, n_block, 
                            cls_layer_list, loc_layer_list, 
                            pred_cls_num, pred_loc_len, 
                            sc_type=None,  gcn_act_fn="relu", 
                            bias=True, concat_feature=False,
                            use_batch_norm=False, dropout_rate=0,
                            use_rd=True, rd_act_fn="relu"):
        super(GCN_Model, self).__init__()

        self.feature_size =feature_size
        self.hidden_size = hidden_size
        self.use_rd = use_rd
        self.readout = None
        self.pred_cls_num = pred_cls_num
        self.pred_loc_len = pred_loc_len
        self.cls_layer_list = cls_layer_list
        self.loc_layer_list = loc_layer_list

        self.relative_feature_fn = sparse_matrix_multiplication

        self.feature_mlp_list = [self.feature_size, 32, 64, 128, 256]
        self.feature_mlp = fc_mlp_layer(self.feature_mlp_list, self.hidden_size, False)

        self.gcn_blocks = nn.ModuleList()
        self.gcn_blocks.append(GraphConvolutionBlock(self.hidden_size, self.hidden_size, sc_type, gcn_act_fn, bias, use_batch_norm, dropout_rate, concat_feature=concat_feature))
        #self.gcn_blocks.append(GraphConvolutionBlock(self.feature_size, self.hidden_size, sc_type, gcn_act_fn, bias, use_batch_norm, dropout_rate, concat_feature=concat_feature))

        for block in range(n_block - 1):
            if sc_type != None and concat_feature:
                self.hidden_size = self.hidden_size * 2
            self.gcn_blocks.append(GraphConvolutionBlock(self.hidden_size, self.hidden_size, sc_type, gcn_act_fn, bias, use_batch_norm, dropout_rate, concat_feature=concat_feature))

        if sc_type != None and concat_feature:
            self.hidden_size = self.hidden_size * 2

        if self.use_rd:
            self.readout = ReadOut(self.hidden_size, self.hidden_size, rd_act_fn)
        
        if sc_type != None and concat_feature:
            self.cls_layer_list = deque(self.cls_layer_list)
            self.cls_layer_list.appendleft(self.hidden_size)
            self.cls_layer_list = list(self.cls_layer_list)

            self.loc_layer_list = deque(self.loc_layer_list)
            self.loc_layer_list.appendleft(self.hidden_size)
            self.loc_layer_list = list(self.loc_layer_list)

        self.predictor = Predictor(self.pred_cls_num, self.pred_loc_len, self.cls_layer_list, self.loc_layer_list)

    def forward(self, adj, adj_relative_local, adj_realative_global, local_feature, global_feature):
        pred_cls = None
        pred_loc = None

        n_feature = global_feature

        global_relative_xyz = self.relative_feature_fn(adj_realative_global, global_feature[:, 0:3])

        n_feature[:, 0:3] = global_relative_xyz

        n_feature= self.feature_mlp(n_feature)
        #n_feature= self.feature_mlp(global_feature)

        for block in self.gcn_blocks:
            n_feature = block(adj, n_feature)
        
        if self.use_rd:
            r_feature = self.readout(n_feature)
            pred_cls, pred_loc = self.predictor(r_feature)
        else:
            pred_cls, pred_loc = self.predictor(n_feature)

        return pred_cls, pred_loc

    def get_prob(self, logits):
        #return F.softmax(logits, dim=1)
        softmax = nn.Softmax(dim=1)
        return softmax(logits)

    def loss(self, pred_cls, labels, pred_loc, gt_box, valid_box,
                    cls_loss_type="focal_sigmoid", loc_loss_type="huber_loss", 
                    cls_loss_weight=1.0, loc_loss_weight=1.0):

        cls_loss_fn = get_loss_type(cls_loss_type)
        loc_loss_fn = get_loss_type(loc_loss_type)

        cls_loss = cls_loss_fn(pred_cls, labels)
        num_cls_point = cls_loss.shape[0]
        cls_loss = cls_loss_weight * cls_loss.mean() 

        box_idx = torch.arange(0, pred_loc.shape[0])
        box_idx = box_idx.unsqueeze(1).to(labels.device)
        box_idx = torch.cat([box_idx, labels], dim=1).long()
        pred_box = pred_loc[box_idx[:, 0], box_idx[:, 1]]

        loc_loss = loc_loss_fn(pred_box, gt_box.squeeze()) * loc_loss_weight
        loc_loss = loc_loss * valid_box.squeeze(1)
        num_valid_point = valid_box.sum()

        m_loc_loss = loc_loss.mean(dim=1)

        if num_valid_point==0:
            loc_loss = 0
        else:
            loc_loss = m_loc_loss.sum() / num_valid_point

        classwise_loc_loss = []

        for cls_idx in range(self.pred_cls_num):
            cls_mask = torch.nonzero(labels==int(cls_idx), as_tuple=False)
            cls_loc_loss = m_loc_loss[cls_mask]

            if cls_loc_loss.shape[0] == 0:
                cls_loc_loss = torch.zeros(1, 2)

            classwise_loc_loss += [cls_loc_loss]

        loss_dict = {}
        loss_dict["classwise_loc_loss"] = classwise_loc_loss

        reg_parameter = torch.cat([param.view(-1) for param in self.parameters()])
        reg_loss = torch.mean(reg_parameter.abs())

        loss_dict.update(
            {
                "cls_loss": cls_loss,
                "loc_loss": loc_loss,
                "reg_loss": reg_loss,
                "num_cls_point": num_cls_point,
                "num_valid_point": num_valid_point
            }
        )

        return loss_dict