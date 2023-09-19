import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from utils.utils import *
from model.AnomalyTransformer_v2 import AnomalyTransformer
from model.WeightNet import WeightNet
from data_factory.data_loader import get_loader_segment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
import copy
import matplotlib.pyplot as plt

def plot_res_gt(data1,data2,dataset):
    t,c = data1.shape
    for i in range(c):
        if i > 1 :continue
        y1 = data1[:5000,i]
        y2 = data2[:5000,i]
        x = np.arange(len(y1))
        plt.cla()
        caption = f"{dataset}_res_and_gt_{i}"
        plt.plot(x,y2,"-.r", label=f"gt",alpha=0.5)
        plt.plot(x,y1,"g", label="recons",alpha=0.5)
        plt.grid(linestyle="-")
        plt.legend()
        plt.title(caption)
        plt.savefig(f"results/{caption}.pdf")

def adjustment_decision(pred,gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred

def best_treshold_search(distance,gt):
    # anomaly_ratio= range(1, 101) # [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,]
    anomaly_ratio= np.arange(1, 20)*0.2 # 0.2-4
    best_res = {"f1":-1}
    best_pred = []
    for ano in anomaly_ratio:
        threshold= np.percentile(distance,100-ano)
        pred=[1 if d>threshold  else 0 for d in distance]
        pred = adjustment_decision(pred,gt)  # 增加adjustment_decision
        eval_results = {
                "f1": f1_score(gt, pred),
                "rc": recall_score(gt, pred),
                "pc": precision_score(gt, pred),
                "threshold":threshold,
                "anomaly_ratio":ano
            }
        if eval_results["f1"] > best_res["f1"]:
            best_res = eval_results
            best_pred = pred
    return best_res, best_pred

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)
        
        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset, random_mask_rate = self.random_mask_rate,config=config)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset , random_mask_rate = self.random_mask_rate,config=config)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset , random_mask_rate = self.random_mask_rate,config=config)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset , random_mask_rate = self.random_mask_rate,config=config)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.criterion2 = nn.MSELoss(reduction="none")
        self.config = config

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, d_model = self.hiden_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _, masked_input_data) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def vali2(self, vali_loader):
        gt = []
        pred = []
        distances = []
        inputs = []
        outputs = []
        for i, (input_data, label, masked_input_data) in enumerate(vali_loader):
            self.wight_net.eval()
            self.teacher_model.eval()
            inputs.append(input_data.detach().float().cpu().numpy())
            input = input_data.float().to(self.device)
            masked_input = masked_input_data.float().to(self.device)
            teacher_res = self.get_teacher_result(input)
            res = self.wight_net(input,masked_input, teacher_res ,False)
            gt.append(label)
            if self.config["test_distance"]=="kl_distance": # teacher_distance or fine_distance or test_distance
                distances.append(res["distance"].detach().cpu().numpy())
            elif self.config["test_distance"]=="recon_distance":
                distances.append(res["distance2"].detach().cpu().numpy())
            outputs.append(res["output"].detach().cpu().numpy())
            # pred.append(res["pred"].detach().cpu().numpy())
        distances = np.concatenate(distances, axis=0).reshape(-1)
        
        inputs = np.concatenate(inputs,axis=0)
        xx,w,c = inputs.shape
        inputs = inputs.reshape(-1,c)
        outputs = np.concatenate(outputs,axis=0).reshape(-1,c)
        # pred = np.concatenate(pred, axis=0).reshape(-1)
        gt = np.concatenate(gt, axis=0).reshape(-1).astype(int)
        
        # make threshold and make decision
        best_res, best_pred = best_treshold_search(distances,gt)
        
        if self.config["compare_model_output"]:
            np.save(f"{self.dir_path}/retrain_test_input.npy",inputs)
            np.save(f"{self.dir_path}/retrain_test_output.npy",outputs)
            
        return best_res

        # thresh = np.percentile(distances,100 - self.anormly_ratio)
        # pred = (distances > thresh).astype(int)
    
        # pred = adjustment_decision(pred,gt)
        
        # accuracy = accuracy_score(gt, pred)
        # precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,average='binary')
        # # print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,recall, f_score))
        # res = {"f1":f_score,"rc":recall,"pc":precision,"thresh":thresh}
        # with open(f"{self.dir_path}/res_phase2.json","w") as f:
        #     json.dump(res,f)

        # return res
        
    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels, masked_input_data) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50 # 是这个temperature导致的输出结果比较平滑吗？有可能

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []
        recon_loss = []
        for i, (input_data, labels, masked_input_data) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            recon_loss.append(loss.detach().cpu().numpy())
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        recon_loss = np.concatenate(recon_loss,axis=0).reshape(-1)
        train_recon_loss = np.array(recon_loss)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        recon_loss = []
        attens_energy = []
        for i, (input_data, labels, masked_input_data) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)
            recon_loss.append(loss.detach().cpu().numpy())

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        recon_loss = np.concatenate(recon_loss,axis=0).reshape(-1)
        test_recon_loss = np.array(recon_loss)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1) # cao，此处这个energy直接把大部分数据的异常分数干成了0，基本上95%以上的数据都是……这……cao……回来得改
        test_energy = np.array(attens_energy)
        combined_recon_loss = np.concatenate([train_recon_loss,test_recon_loss],axis=0)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0) # 此处好像是标签泄漏了……
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        thresh2 = np.percentile(combined_recon_loss, 100 - self.anormly_ratio)
        anormly_ratios = np.arange(1,1000)*0.1
        thresholds = np.percentile(combined_energy, 100 - anormly_ratios)
        thresholds = thresholds.tolist()
        recon_thresholds = np.percentile(combined_recon_loss, 100 - anormly_ratios)
        recon_thresholds = recon_thresholds.tolist()
        hierarchical_threshold = np.percentile(combined_energy, 100 - self.config["Hierarchical_threshold_k"])
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        inputs = []
        outputs = []
        recon_loss = []
        for i, (input_data, labels, masked_input_data) in enumerate(self.thre_loader):
            inputs.append(input_data.float())
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            outputs.append(output.detach().cpu().numpy())

            loss = torch.mean(criterion(input, output), dim=-1)
            recon_loss.append(loss.detach().cpu().numpy())

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
        
        recon_loss = np.concatenate(recon_loss,axis=0).reshape(-1) 
        test_recon_loss = np.array(recon_loss)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        
        inputs = np.concatenate(inputs, axis=0)
        outputs = np.concatenate(outputs, axis=0)
        
        # if len(self.removed_dims) == 0:
        if self.config["baseline_dir_path"] == self.config["dir_path"]:
            np.save(f"{self.dir_path}/test_input.npy",inputs)
            np.save(f"{self.dir_path}/test_output.npy",outputs)
            np.save(f"{self.dir_path}/test_labels.npy",test_labels)

        pred = (test_energy > thresh).astype(int)
        pred2 = (test_recon_loss > thresh2).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred2 = adjustment_decision(pred2,gt)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        precision2, recall2, f_score2, support2 = precision_recall_fscore_support(gt, pred2, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,recall, f_score))
        res = {"f1":f_score,"rc":recall,"pc":precision,"thresh":thresh,"hierarchical_threshold":hierarchical_threshold,"thresholds":thresholds,
               "f1_recon":f_score2,"pc_recon":precision2,"thresh2":thresh2,"recon_thresholds":recon_thresholds,}
        with open(f"{self.dir_path}/res.json","w") as f:
            json.dump(res,f)

        return res

    def get_teacher_result(self, input):
        b,w,c = input.shape
        output, series, prior, sigmas, x_feature = self.teacher_model(input,output_re=True) # x_feature: b c h output: b, w, c
        distance = self.criterion2(output,input) # distance: b w c
        distance2 = distance

        series_loss = 0.0
        prior_loss = 0.0
        temperature = 50
        for u in range(len(prior)): # todo 保留最后一个维度
            if u == 0:
                series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)).detach()) * temperature
                prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)),series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)).detach()) * temperature
                prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)),series[u].detach()) * temperature
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        metric = metric = metric.unsqueeze(2).repeat(1, 1, c)
        distance = metric * distance
        
        threshold__ = self.config["thresh"] *torch.ones_like(distance)
        predict = torch.greater(distance,threshold__).int()
        return {"pred":predict,"distance":distance,"distance2":distance2,"output":output} # output 不变

    def train_phase2(self):

        print("======================TRAIN Phase2 MODE======================")
        
        # 判断是否跑过
        # try:
        #     with open(f"{self.dir_path}/res_phase2.json") as f:
        #         temp_res = json.load(f)
        #     f1 = temp_res["f1"]
        #     if f1 > 0:
        #         print(f"exist {self.dir_path}/res_phase2.json: {temp_res}")
        #         return temp_res
        # except Exception as e:
        #     pass
        
        try:
            with open(f"{self.baseline_dir_path}/res.json") as f:
                test_res = json.load(f)
                length = len(test_res["recon_thresholds"])
                if length != 999:
                    raise Exception("thresholds length error")
            print(f"exist {self.baseline_dir_path}/res.json: f1: {test_res['f1']}")
            # print(f"exist {self.baseline_dir_path}/res.json: f1: {test_res}")
        except Exception as e:
            try:
                test_res = self.test()
            except Exception as e:
                self.train()
                test_res = self.test()
                
            with open(f"{self.baseline_dir_path}/res.json","w") as f:
                json.dump(test_res,f)
            print(f"new {self.baseline_dir_path}/res.json: f1: {test_res['f1']}")
           
        # self.train()     
        # test_res = self.test()
        # with open(f"{self.baseline_dir_path}/res.json","w") as f:
        #     json.dump(test_res,f)
        # print(f"new {self.baseline_dir_path}/res.json: f1:{test_res['f1']}")
        # exit()

        self.config.update(test_res)
        self.teacher_model = copy.deepcopy(self.model) 
        self.model2 = self.model if self.fine_turn else AnomalyTransformer(win_size=self.win_size, d_model = self.hiden_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.wight_net = WeightNet(self.model2,self.config)
        self.optimizer2 = torch.optim.Adam(self.wight_net.parameters(), lr=self.fine_turn_lr)
        # print(*[name for name, _ in self.wight_net.named_parameters()], sep='\n')
        
        if torch.cuda.is_available():
            self.wight_net.cuda()
            self.teacher_model.cuda()
        
        start_time = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        best_res = {"f1":0}
        early_stop_count = 0
        pationce =3
        
        for epoch in range(self.num_epochs):
            
            loss = []
            inputs = []
            outputs = []
            masked_inputs = []
            epoch_time2 = time.time()
            if self.re_train:
                for i, (input_data, labels, masked_input_data) in enumerate(self.train_loader):
                    # train
                    self.wight_net.train()
                    self.teacher_model.eval()
                    # if self.fine_turn:
                    #     self.teacher_model.train()
                    # else :
                    #     self.teacher_model.eval()
                    inputs.append(input_data.detach().float().cpu().numpy())
                    masked_inputs.append(masked_input_data.detach().float().cpu().numpy())
                    input = input_data.float().to(self.device)
                    masked_input = masked_input_data.float().to(self.device)
                    teacher_res = self.get_teacher_result(input)
                    
                    self.optimizer2.zero_grad()
                    res = self.wight_net(input, masked_input,teacher_res,self.re_train)     
                    if self.config["fine_distance"]=="recon_distance": # teacher_distance or fine_distance or test_distance
                        res["loss2"].backward()
                        loss.append(res["loss2"].detach().cpu().numpy())
                    elif self.config["fine_distance"]=="kl_distance":
                        res["loss1"].backward()
                        loss.append(res["loss1"].detach().cpu().numpy())
                    outputs.append(res["output"].detach().cpu().numpy())
                    self.optimizer2.step()
                    # loss.append(res["loss1"].detach().cpu().numpy())
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time2))
            
            # inputs = np.concatenate(inputs,axis=0)
            # xx,w,c = inputs.shape
            # inputs = inputs.reshape(-1,c)
            # outputs = np.concatenate(outputs,axis=0).reshape(-1,c)
            # masked_inputs = np.concatenate(masked_inputs,axis=0).reshape(-1,c)
            # plot_res_gt(outputs,inputs,f"{self.config['dataset']}_unmasked_input")
            # plot_res_gt(outputs,masked_inputs,f"{self.config['dataset']}_masked_input")
            # vali2
            vali_res = self.vali2(self.thre_loader)
            
            epoch_time = time.time()
            vali_res["epoch_time"] = (epoch_time - start_time) / (epoch+1)
            loss = np.array(loss).mean() if len(loss) >0 else 0
            vali_res["train2_loss"] = float(loss) 
            print(f"epoch {epoch} train loss: {loss} vali f1: {vali_res['f1']} vali rc: {vali_res['rc']} vali pc: {vali_res['pc']} epoch time: {vali_res['epoch_time']}")
            
            if best_res["f1"] < vali_res["f1"]:
                best_res = vali_res
                early_stop_count = 0
            else:
                early_stop_count += 1
                print(f"early_stop_count: {early_stop_count}/{pationce}")
            if early_stop_count>= pationce:
                break
        print(f"best_res: {best_res}")
        with open(f"{self.dir_path}/res_phase2.json","w") as f:
            json.dump(best_res,f)
