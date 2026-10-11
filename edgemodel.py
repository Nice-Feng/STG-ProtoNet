import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from layers import GraphConvolution
from torch import nn

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.01, lrpatience=5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.counter_lr = 0
        self.min_validation_loss = float('inf')
        self.min_validation_loss_lr = float('inf')
        self.lrpatience = lrpatience
        self.minlr = 0.0001

    def early_stop(self, validation_loss, lr):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if int(self.counter >= self.patience) | int(self.minlr >= lr):
                return True
        return False

    def decrease_lr(self, validtion_loss, lr):
        if validtion_loss < self.min_validation_loss_lr:
            self.min_validation_loss_lr = validtion_loss
            self.counter_lr = 0
        elif validtion_loss > (self.min_validation_loss_lr):
            self.counter_lr += 1
            if self.counter_lr >= self.lrpatience:
                lr = lr * 0.5
        return lr
class PowerLayer(nn.Module):

    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))

class lunwen4(torch.nn.Module):
    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph,num_prototypes):
        # input_size: EEG frequency x channel x datapoint
        super(lunwen4, self).__init__()
        self.window = [0.5, 0.25, 0.125]
        self.pool = pool
        self.GCN = GraphConvolution(19, out_graph)
        self.global_adj = nn.Parameter(torch.FloatTensor(30, 30), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        self.bn = nn.BatchNorm1d(30)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_t2 = nn.BatchNorm2d(int(num_T*0.5))
        self.BN_t3 = nn.BatchNorm1d(112)
        self.num_prototypes=num_prototypes;
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1,int( self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.last_layer = nn.Linear(self.num_prototypes,2,
                                    bias=False)  # do not use bias
        self.prototype_shape=[10,112]
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, int(num_T*0.5), kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.AvgPool2d((2, 2)))
        self.epsilon = 1e-4
        self.Softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(19, 32)

    def forward(self, x,y,edge):
        adj = self.get_adj(edge)
        out = self.bn(x.float())

        out = self.GCN(out.float(), adj.float())
        xv2=x.unsqueeze(1).float()
        y = self.Tception1(self.linear(xv2))
        out1=y
        y = self.Tception2(out)
        out1 = torch.cat((out1, y), dim=-1)
        y = self.Tception3(out)
        out1 = torch.cat((out1, y), dim=-1)

        out1 = self.BN_t(out1)
        out1 = self.OneXOneConv(out1)
        out1 = self.BN_t2(out1)
        out1 = torch.reshape(out1, (out1.size(0),  -1))
        out1= self.BN_t3(out1)
        prototype_activations, min_distances = self.prototype_distances(out1)
        logits = self.last_layer(prototype_activations)
        probs = self.Softmax(logits)
        return probs, prototype_activations

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_adj(self, x, self_loop=True):
        # x: b, node, feature
        B2, N, _ = x.shape
        adj_list = []
        for b in range(B2):
            out1 = x[b]
            B, _ = out1.shape
            # 1. Min‑Max归一化到0‑1，每个样本独立
            flatten = out1.reshape(1, -1)  # [B, N*N]
            min_v, _ = flatten.min(dim=1, keepdim=True)
            max_v, _ = flatten.max(dim=1, keepdim=True)
            flatten_norm = (flatten - min_v) / (max_v - min_v + 1e-8)
            out_norm = flatten_norm.reshape(B,B)  # [B,N,N], 值域0~1
            threshold = torch.quantile(flatten_norm, q=0.2, dim=1, keepdim=True)  # [B,1]
            mask = flatten_norm >= threshold
            flatten_sparse = flatten_norm * mask.float()
            out_sparse = flatten_sparse.reshape(B, B)
            adj = (out_sparse + out_sparse.transpose(-1, -2)) / 2.0
            num_nodes = adj.shape[-1]
            adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
            if self_loop:
                adj = adj + torch.eye(num_nodes, device=adj.device)
            rowsum = torch.sum(adj, dim=-1)
            mask = torch.zeros_like(rowsum)
            mask[rowsum == 0] = 1
            rowsum += mask
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
            adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
            adj_list.append(adj)
        return torch.stack(adj_list, dim=0)

    def self_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s

    def prototype_distances(self, x):
        x_min, _ = torch.min(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)

        x_min, _ = torch.min(self.prototype_vectors, dim=1, keepdim=True)
        x_max, _ = torch.max(self.prototype_vectors, dim=1, keepdim=True)
        prototype_vectors_norm = (self.prototype_vectors - x_min) / (x_max - x_min + 1e-8)

        xp = torch.mm( x_norm, torch.t(prototype_vectors_norm))
        distance = -2 * xp + torch.sum( x_norm ** 2, dim=1, keepdim=True) + torch.t(
            torch.sum(prototype_vectors_norm ** 2, dim=1, keepdim=True))
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        return similarity, distance
# =====新增这一段=====
    def get_graph_embedding(self, x, y, edge_all):
        _, emb = self.forward(x, y, edge_all)
        return emb


class Aggregator():

    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area
        self.chan_in_area = idx_area
        self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)

    def forward(self, x):
        # x: batch x channel x data
        data = []
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
            else:
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)

    def get_idx(self, chan_in_area):
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)
        return idx_[1:]

    def aggr_fun(self, x, dim):
        # return torch.max(x, dim=dim).values
        return torch.mean(x, dim=dim)

