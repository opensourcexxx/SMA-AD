import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def my_kl_loss(p, q): # b head w w 
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1) # b head

class WeightNet(nn.Module):
    def __init__(self, model, config):
        super(WeightNet, self).__init__()
        # self.weight = nn.Linear(window_size, 1,bias=False)
        padding=1 if torch.__version__>="1.5.0" else 2
        self.win_size = config["win_size"] 
        self.hidden_size = config["hiden_size"]
        self.k = config["k"]
        self.base_model = model
        self.config = config
        self.pseudo_label_type = config["pseudo_label_type"]
        self.recon_mask_type = config["recon_mask_type"]
        self.criterion = nn.MSELoss(reduction='none')
        self.tanh = nn.Tanh()
        
    def get_distance_from_anomaly_transform(self, x):
        b,w,c = x.shape
        output, series, prior, sigmas, x_feature = self.base_model(x,output_re=True) # x_feature: b c h output: b, w, c
        distance = self.criterion(output,x).mean(-1) # distance: b w c
        distance2 = distance

        series_loss = 0.0
        prior_loss = 0.0
        temperature = 50 # 
        for u in range(len(prior)): # 
            if u == 0:
                series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)).detach()) * temperature
                prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)),series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)).detach()) * temperature
                prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)),series[u].detach()) * temperature
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        # metric = metric.unsqueeze(2).repeat(1, 1, c)
        distance = metric * distance
        
        return output,distance,distance2
        
    def get_masked_weight(self,x):
        b,w,c = x.shape
        x_ = x.mean(dim=-2)
        thresh_0 = torch.tensor(0)*torch.ones_like(x_)
        thresh_00001 = torch.tensor(0.0001)*torch.ones_like(x_)
        temp1 = torch.logical_and(torch.greater(thresh_00001,x_),torch.ge(x_,thresh_0)).int() 
        temp2 = torch.logical_and(torch.greater(x_,-thresh_00001),torch.ge(thresh_0,x_)).int() 
        masked_weight = torch.logical_or(temp1,temp2).int() 
        masked_weight = masked_weight.unsqueeze(dim=1).repeat(1,w,1)
        return masked_weight
    
    def get_train_loss_from_anomaly_transform_v2(self, x, masked_x, pseudo_label):
        masked_weight = self.get_masked_weight(masked_x)
        b,w,c = x.shape
        if self.recon_mask_type == "none":
            output, series, prior, sigmas, x_feature = self.base_model(x,output_re=True) # x_feature: b c h output: b, w, c
            distance = self.criterion(output,x) 
        elif self.recon_mask_type == "recon_unmasked":
            output, series, prior, sigmas, x_feature = self.base_model(masked_x,output_re=True) # x_feature: b c h output: b, w, c
            distance = self.criterion(output,masked_x) 
            distance = distance*(1-masked_weight)
        elif self.recon_mask_type == "recon_all":
            output, series, prior, sigmas, x_feature = self.base_model(masked_x,output_re=True) # x_feature: b c h output: b, w, c
            distance = self.criterion(output,x) 
        elif self.recon_mask_type == "recon_masked":
            output, series, prior, sigmas, x_feature = self.base_model(masked_x,output_re=True) # x_feature: b c h output: b, w, c
            distance = self.criterion(output,x-masked_x) 
            distance = distance*masked_weight
        # distance: b w c
        
        if self.pseudo_label_type == 'self_softmax':
            pseudo_label = torch.softmax(distance,dim=-1) + 1
        elif self.pseudo_label_type == 'focal_loss':
            pseudo_label = 1 - distance
        elif self.pseudo_label_type == 'invers_focal_loss':
            pseudo_label = 1 + distance
        
        # add weight network
        # weight = self.weight2(output.permute(0,2,1)).permute(0,2,1)
        # weight = self.tanh(weight)*self.weight_scope+1
        distance = distance*pseudo_label # 区别1
        distance2 = distance.mean()
        # distance = weight.repeat(1,self.win_size,1)*distance
        # distance = distance.mean()
        
        # calculate Association discrepancy
        series_loss = 0.0
        prior_loss = 0.0
        temperature = 1 # 区别2
        for u in range(len(prior)): # 
            if u == 0:
                series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)).detach()) * temperature
                prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)),series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)).detach()) * temperature
                prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)),series[u].detach()) * temperature
        # todo check series_loss 和 prior_loss 是否都是负值？
        # 如果他们是正值呢？毕竟kl散度是用来度量两个
        metric = torch.softmax((-series_loss - prior_loss), dim=-1) # 直接让两个注意力的分布的kl散度尽量的小不如让两个注意力分布差异比较大的点重构误差尽量的小……
        # 其实还是在帮助训练，帮助模型重构难以重构的点。这其实有点反直觉，就是这些难以重构的点需要更大的力度来学习重构。可能模型还没有完全的学到如何去重构正常数据的分布！
        # 或许应该试试，给难以重构的数据以更多的损失权重，是否会有奇效？
        # 好像不对，两个kl散度去负数之后，就是kl散度越大，其对应的权重越小，那么就是让模型少去重构难重构的点，多去重构容易重构的点！
        # 和我们思想是一样的，不过他是使用分布差异的角度来做的。不对，kl散度越小，代表约可能是异常，因为prior默认关注周边，如果series也关注周边，那么就是说明此处可能是异常，此处不容易重构。
        # 因此之前的分析是对的，此处的Metric就是在帮助模型去学习如何重构难以重构的点
        # 所以需要去试试是否有奇效
        metric = metric.unsqueeze(2).repeat(1, 1, c)
        distance = metric * distance # 此处distance乘上metric之后其实就已经两级分化了，此时再进行softmax没有太大意义，所以softmax一定是运行在recon_distance上的
        distance = distance.mean()

        return output,distance,distance2
        
    def forward(self, x, masked_x, teacher_res, re_train=True): # b, w, c  
        if self.config["teacher_distance"] == "kl_distance": # Hierarchical_threshold_k 1-10
            thresholds = self.config["thresholds"]  # kl_thresholds
            teacher_distance = teacher_res["distance"] # kl_distance
            threshold = self.config["thresh"] *torch.ones_like(teacher_distance)  # 前1%
        elif self.config["teacher_distance"] == "recon_distance": # Hierarchical_threshold_k 1-99
            thresholds = self.config["recon_thresholds"] # recon_thresholds
            teacher_distance = teacher_res["distance2"] # recon_distance
            threshold = self.config["thresh2"]  *torch.ones_like(teacher_distance) # 前1%
            
        normal = torch.greater(threshold,teacher_distance).int()
        abnormal = torch.greater(teacher_distance,threshold).int()
        if re_train:
            # self. # batch 算出的 threshold 可能不靠谱
            if self.pseudo_label_type == 'none':
                pseudo_label = 1
            elif self.pseudo_label_type == "min_normal":
                pseudo_label = normal
            elif self.pseudo_label_type == "min_normal_max_abnormal":
                pseudo_label = normal - abnormal
            elif self.pseudo_label_type == "max_abnormal": # 这种就没必要存在了，哪有只扩大重构误差的，这不是强行让模型训外吗
                pseudo_label = - abnormal
            elif self.pseudo_label_type == "min_abnormal":
                pseudo_label = abnormal
            elif self.pseudo_label_type == "min_normal_double_min_abnormal":
                pseudo_label = normal + 2*abnormal
            elif self.pseudo_label_type == "soft_min_abnormal":
                pseudo_label = torch.softmax(teacher_distance, dim=-1) + 1
            elif self.pseudo_label_type == "threshold_k_min_abnormal": #只有下方两个涉及到利用teacher_distance重进计算标签
                threshold = torch.quantile(teacher_distance,torch.tensor([self.config["threshold_k"]]).to(teacher_distance.device))*torch.ones_like(teacher_distance)
                normal = torch.greater(threshold,teacher_distance).int()
                abnormal = torch.greater(teacher_distance,threshold).int()
                pseudo_label = abnormal +1 # todo retest threshold_k_min_abnormal
            elif self.pseudo_label_type == "Hierarchical_pseudo_label":
                threshold = thresholds[int(self.config["Hierarchical_threshold_k"]*10)] *torch.ones_like(teacher_distance)
                normal2 = torch.greater(threshold,teacher_distance).int()
                # hard2recons = normal2 - normal
                hard2recons = normal - normal2
                pseudo_label = normal + hard2recons*2 - abnormal # 这个2倍是否真的有用？
            else:
                pseudo_label = 1
                
            output, loss1, loss2 = self.get_train_loss_from_anomaly_transform_v2(x, masked_x,pseudo_label)
            return {"output":output,"loss1":loss1,"loss2":loss2}
        
        else:
            output, distance, distance2 = self.get_distance_from_anomaly_transform(x)
            # threshold_ = self.config["thresh"] *torch.ones_like(distance)
            # predict = torch.greater(distance,threshold_).int()
            return {"distance":distance,"distance2":distance2,"output":output} # output 不变

        
        
    # def get_train_loss_from_anomaly_transform(self, x, pseudo_label):
    #     output, series, prior, sigmas, x_feature = self.base_model(x,output_re=True) # x_feature: b c h output: b, w, c
    #     distance = self.criterion(output,x) # distance: b w c
        
    #     # add weight network
    #     # weight = self.weight2(output.permute(0,2,1)).permute(0,2,1)
    #     # weight_ = self.tanh(weight)*self.weight_scope+1
    #     # distance = distance*pseudo_label
    #     # distance = weight.repeat(1,self.win_size,1)*distance
    #     distance = distance.mean()
        
    #     # calculate Association discrepancy
    #     series_loss = 0.0
    #     prior_loss = 0.0
    #     for u in range(len(prior)): # (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)) 是为了让prior[u]归一化
    #         series_loss += (torch.mean(my_kl_loss(series[u], 
    #                                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)).detach())) 
    #                         + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)).detach(),
    #                                                 series[u])))
    #         prior_loss += (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)),
    #                                              series[u].detach())) 
    #                        + torch.mean(my_kl_loss(series[u].detach(), 
    #                                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,self.win_size)))))
    #     series_loss = series_loss / len(prior)
    #     prior_loss = prior_loss / len(prior)
    #     distance1 = distance - self.k * series_loss
    #     distance2 = distance + self.k * prior_loss

    #     return output,distance1,distance2