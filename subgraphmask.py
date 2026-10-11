import torch
import torch.nn as nn
import torch.nn.functional as F


class subgraphmask(nn.Module):
    def __init__(self, node_features_num, edge_features_num, device):
        super(subgraphmask, self).__init__()
        self.device = device
        self.node_features_num = node_features_num
        self.edge_features_num = edge_features_num
        self.mseloss = torch.nn.MSELoss()
        self.feature_size = 64
        self.linear = nn.Linear(self.node_features_num, self.feature_size).to(self.device)
        self.linear1 = nn.Linear(2 * self.feature_size, 8).to(self.device)
        # self.linear1 = nn.Linear(self.node_features_num * 2, 1).to(self.device)
        self.linear2 = nn.Linear(8, 1).to(self.device)
        a=torch.rand(30, 30)
        self.edge_prob_matrix = (nn.Parameter(a + a.transpose(0, 1))).to(self.device)
        b=torch.rand(30, 30)
        self.edge_prob_matrix_noise= (nn.Parameter(b + b.transpose(0, 1))).to(self.device)
    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()) + (1 - bias)
            gate_inputs = (torch.log(eps) - torch.log(1 - eps)).to(self.device)
            gate_inputs = gate_inputs.to(self.device)
            # print(f'\ntemperature{temperature.device}')
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """
        Sample from the instantiation of concrete distribution when training
        \epsilon \sim  U(0,1), \hat{e}_{ij} = \sigma (\frac{\log \epsilon-\log (1-\epsilon)+\omega_{i j}}{\tau})
        """
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def _edge_prob_mat(self, graph):
        # The number of nodes in the graph: node_num
        # the number of features in the graph: feature_num

        # x = graph.type(dtype=torch.float32)

        edgemask = self.edge_prob_matrix[graph.edge_index[0], graph.edge_index[1]]
        edgemask_noise = self.edge_prob_matrix_noise[graph.edge_index[0], graph.edge_index[1]]
        edgemask = self._sample_graph(edgemask, temperature=0.5, bias=0.0, training=self.training)
        edgemask_noise  = self._sample_graph(edgemask_noise, temperature=0.5, bias=0.0, training=self.training)
        return edgemask,edgemask_noise

    def forward(self, graph):
        grapuse=graph
        graphnoise=graph
        pos_penalty=0
        neg_penalty=0
        for i in range(len(graph)):
            subgraph = graph[i]
            edge_prob_matrix,edge_prob_matrix_noise = self._edge_prob_mat(subgraph)
            graph[i].attr = edge_prob_matrix
            graphnoise[i].attr= edge_prob_matrix_noise
            pos_penalty += edge_prob_matrix.var()
            neg_penalty += edge_prob_matrix_noise.var()

        return graph, graphnoise, pos_penalty,neg_penalty

def pairwise_distances(x):
    # x should be two dimensional
    # The latter dimension of x is the number of features of the graph embedding after GNN
    x = x.view(-1, 1280)
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()
def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    # dist = dist/torch.max(dist)
    return torch.exp(-dist / sigma)


def renyi_entropy(x, sigma):
    """
    Function for computing matrix entropy.
    """
    alpha = 5
    k = calculate_gram_mat(x, sigma)
    k = k / torch.trace(k)
    # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eigv = torch.abs(torch.linalg.eigh(k, UPLO='U')[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y):
    """
    Function for computing joint matrix entropy.
    """
    alpha = 5
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y)
    k = k / torch.trace(k)
    # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eigv = torch.abs(torch.linalg.eigh(k, UPLO='U')[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))

    return entropy


def calculate_MI(x, xnoise, s_x, s_xnoise):
    """
    Function for computing mutual information using matrix entropy
    """
    Hx = renyi_entropy(x, sigma=s_x)
    Hxnoise = renyi_entropy(xnoise, sigma=s_xnoise)
    Hxy = joint_entropy(x, xnoise, s_x, s_xnoise)
    Ix_xnoise = Hx + Hxnoise - Hxy
    # normlize = Ixy/(torch.max(Hx,Hy))


    return Ix_xnoise