import numpy as np
import torch
import os
import scipy.io as sio
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from edgemodel import lunwen4
import torch.nn as nn
from torch_geometric.loader import DataLoader
from edgemodel import EarlyStopper
import csv
from subgraphmask import subgraphmask
import random
import copy
from openpyxl import Workbook

class REC(nn.Module):
    def adjust_lr(self, epoch):
        self.lr = 0.01 * (0.1 ** (epoch // 8))

    def load_module(self, num_chan_local_graph,device,ind):
        self.edgemodel = lunwen4(num_classes=2,  input_size=(1, 32, 512), sampling_rate=19,
                           num_T=16, out_graph=32,pool=16,pool_step_rate=0.25,idx_graph=num_chan_local_graph,
                           dropout_rate=0.5,num_prototypes=10).to(device)
        self.lr = 0.001
        self.params = list(self.edgemodel.parameters()) #+ list(self.gin_layer.parameters()) + list( self.class_model.parameters())
        self.subgraphmask = subgraphmask(node_features_num=19, edge_features_num=4, device=device)
        self.optimizer = torch.optim.Adam([
            {'params': self.edgemodel.parameters(), 'lr': 0.001},
            {'params': self.subgraphmask.parameters(), 'lr': 0.001}
        ])
        self.checkpoint=[]
        project_root = os.path.abspath("outs")
        self.criterion=nn.CrossEntropyLoss()
        self.bestevalacc=0
        self.besttestacc=0
        self.device=device
        self.mi_weight = 0.01
        self.bestmodel = None
        self.outdir = os.path.abspath("bestmodel_aftertrain")
        os.makedirs(self.outdir, exist_ok=True)
        self.ind = ind

    def train_data(self, epochnum, device, train_loader, vali_loader, test_loader, early_stopper,sub,batch_size):
        train_acc_all = []
        vali_acc_all = []
        test_acc_all=[]
        eval_acc_all=[]
        for epoch in range(epochnum):
            # self.adjust_lr(epoch)
            loss_all = 0
            num = 0
            acc_accum=0
            for batch in train_loader:
                data_list = batch.to_data_list()
                xdata = []
                ydata = []
                edgedata = []
                for g in data_list:
                    xdata.append(g.x.unsqueeze(0))
                    ydata.append(g.y.unsqueeze(0))
                    edgedata.append(g.edge_attr.unsqueeze(0))
                x= torch.cat(xdata, dim=0).to(device)
                y = torch.cat(ydata, dim=0).to(device)
                edge_all= torch.cat(edgedata, dim=0).to(device)
                y_out,_ = self.edgemodel(x,y,edge_all)
                y = torch.squeeze(y)
                y = y.long()
                loss = self.criterion(y_out, y)
                # optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_all += loss.data.cpu().detach().numpy()
                pred = y_out.max(1, keepdim=True)[1]
                correct = pred.eq(y.view_as(pred)).sum().cpu().item()
                acc = correct / float(len(y))
                acc_accum = acc_accum + acc
                num = num + 1
            acc_train = acc_accum / num
            print("Loss: %f" % (loss_all))
            print("Accuracy train: %f" % (acc_train))
            print('Training epoch: {}'.format(epoch))
            acc_accum=0
            num=0
            self.lr = early_stopper.decrease_lr(loss_all, self.lr)
            print('Learning rate: {}'.format(self.lr))
            print('Loss:{}'.format(loss_all))
            train_acc_all = np.append(train_acc_all, acc_train)
            acc_eval,loss_eval = self.evaluate_data(vali_loader)
            print("accuracy evaluation: %f" % (acc_eval))
            # judge best from epochs
            if acc_eval > self.bestevalacc:
                self.bestevalacc = acc_eval
            acc_test = self.test(test_loader)
            print("accuracy test: %f" % (acc_test))
            # judge best from epochs
            if acc_test > self.besttestacc:
                self.besttestacc = acc_test
                self.bestmodel = copy.deepcopy(self.edgemodel)
            test_acc_all = np.append(test_acc_all, acc_test)
            eval_acc_all = np.append(eval_acc_all, acc_eval)
            stop_flag = early_stopper.early_stop(validation_loss=loss_eval, lr=self.lr)
            if stop_flag:
                print(f"Early Stop Triggered at epoch {epoch}")
                break
        save_path = os.path.join(self.outdir, f"sub{sub}_fold{self.ind}_best.pth")
        torch.save(self.bestmodel.state_dict(), save_path)
        return train_acc_all, test_acc_all, eval_acc_all, self.bestevalacc, self.besttestacc


    def evaluate_data(self, vali_loader):
        acc_accum = 0
        num = 0
        loss_all=0
        for batch in test_loader:
            data_list = batch.to_data_list()
            xdata = []
            ydata = []
            edgedata = []
            for g in data_list:
                xdata.append(g.x.unsqueeze(0))
                ydata.append(g.y.unsqueeze(0))
                edgedata.append(g.edge_attr.unsqueeze(0))
            x = torch.cat(xdata, dim=0).to(device)
            y = torch.cat(ydata, dim=0).to(device)
            edge_all = torch.cat(edgedata, dim=0).to(device)
            y_out,_= self.edgemodel(x, y, edge_all)
            y = torch.squeeze(y)
            y = y.long()
            loss = self.criterion(y_out, y)
            loss_all += loss.data.cpu().detach().numpy()
            pred = y_out.max(1, keepdim=True)[1]
            correct = pred.eq(y.view_as(pred)).sum().cpu().item()
            acc = correct / float(len(y))
            acc_accum = acc_accum + acc
            num = num + 1
        acc_eval = acc_accum / num
        return acc_eval,loss_all


    def test(self, test_loader):
        acc_accum = 0
        num = 0
        for batch in test_loader:
            data_list = batch.to_data_list()
            xdata = []
            ydata = []
            edgedata = []
            for g in data_list:
                xdata.append(g.x.unsqueeze(0))
                ydata.append(g.y.unsqueeze(0))
                edgedata.append(g.edge_attr.unsqueeze(0))
            x = torch.cat(xdata, dim=0).to(device)
            y = torch.cat(ydata, dim=0).to(device)
            edge_all = torch.cat(edgedata, dim=0).to(device)
            y_out,_= self.edgemodel(x, y, edge_all)
            y = torch.squeeze(y)
            y = y.long()
            pred = y_out.max(1, keepdim=True)[1]
            correct = pred.eq(y.view_as(pred)).sum().cpu().item()
            acc = correct / float(len(y))
            acc_accum = acc_accum + acc
            num = num + 1
        acc_eval = acc_accum / num
        return acc_eval


if __name__ == '__main__':
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    original_order = ch_names = ['FP1', 'FP2', 'F11', 'F7', 'F3', 'FZ', 'F4', 'F8', 'F12', 'FT11',
                                 'FC3', 'FCZ', 'FC4', 'FT12', 'T7', 'C3', 'CZ', 'C4', 'T8', 'CP3',
                                 'CPZ', 'CP4', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2']

    # Define proper channel orders for the local-global graphs in LGGNet. Please refer to three graph definitions (general, frontal, hemesphere).
    graph_general_DEAP = [['FP1', 'FP2'], ['F11', 'F7', 'F3'], ['FZ','FCZ','CZ'],  ['F4', 'F8', 'F12'], ['FT11',
                                 'FC3'], ['FC4', 'FT12'], ['T7', 'C3'],  ['C4', 'T8'], ['CP3','P3','P7'],
                                 ['CPZ', 'PZ'],['CP4', 'P4', 'P8'], ['O1', 'OZ', 'O2']]
    data_dir = os.path.join(os.getcwd(), "EEG_data")
    graph_idx = graph_general_DEAP  # The general graph definition for DEAP is used as an example.
    idx = []
    num_chan_local_graph = []

    for i in range(len(graph_idx)):
        num_chan_local_graph.append(len(graph_idx[i]))
        for chan in graph_idx[i]:
            idx.append(original_order.index(chan))

    for sub in list(range(1, 3)):
        tname= os.path.join(data_dir, 'sub' + str(sub), 'xunlianqian', f'rest_sub{sub}.mat')
        task = sio.loadmat(tname)
        data = task['restdata']
        edge = task['restedge']
        label = task['restlabel']-1
        gdata_list = []
        f_1 = open('./acc_train_sub' + str(sub) + '.csv', 'w', encoding='utf8', newline='')
        train_writter = csv.writer(f_1)
        f_2 = open('./acc_validation_sub' + str(sub) + '.csv', 'w', encoding='utf8', newline='')
        eval_writter = csv.writer(f_2)
        f_3 = open('./acc_test_sub' + str(sub) + '.csv', 'w', encoding='utf8', newline='')
        test_writter = csv.writer(f_3)
        for sp in range(0, edge.shape[0]):
            edge_tmp = np.mean(edge[sp,:,:,:], axis=2)
            n_node = edge_tmp.shape[0]
            upper_mask = np.triu(np.ones_like(edge_tmp, dtype=bool), k=0)
            upper_vals = edge_tmp[upper_mask] # 取出上三角全部元素
            threshold = np.percentile(upper_vals,20)# 取50%分位数，大于等于该分位数保留，其余置0
            edge_out = np.zeros_like(edge_tmp)# 复制原矩阵，先全部置0
            #
            edge_out[upper_mask] = np.where(upper_vals >= threshold, upper_vals, 0) #上三角位置，大于等于阈值才赋值
            y_value = int(label[sp].item())
            gdata = Data(x=torch.tensor(data[sp,idx,:]), y=torch.tensor(y_value),edge_attr=torch.tensor(edge_out))
            gdata_list.append(gdata)

        kfold=KFold(n_splits=5,shuffle=True,random_state=1)
        ind=0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for trainind ,test_ind in kfold.split(gdata_list):
            mod = REC()
            mod = mod.to(device)
            early_stopper = EarlyStopper(patience=5, min_delta=0.001, lrpatience=5)
            mod.load_module(idx, device,ind)
            train_val_split = int(len(trainind) * 0.9)
            train_ind_v2 = trainind[:train_val_split]
            val_ind = trainind[train_val_split:]

            train_dataset = [gdata_list[int(i)] for i in train_ind_v2]
            eval_dataset = [gdata_list[int(i)] for i in val_ind]
            test_dataset = [gdata_list[int(i)] for i in test_ind]

            batch_size = 16
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            vali_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            train_acc_all, test_acc_all, eval_acc_all, bestevalacc, besttestacc= mod.train_data(epochnum=50,device=device, train_loader=train_loader, vali_loader=vali_loader,
                test_loader=test_loader,early_stopper=early_stopper,sub=sub,batch_size=8)
            train_writter.writerow([train_acc_all])
            test_writter.writerow([test_acc_all])
            eval_writter.writerow([eval_acc_all])
            ind+=1
        f_1.close()
        f_2.close()
        f_3.close()
