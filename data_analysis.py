from main_for_minmax import*
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
        # scaler.data_range_
    data = scaler.transform(data)
    # print("Data normalized")

    return data, scaler

def get_res(config):
    config.dir_path = config.dir_path = get_save_dir(config)
    res_file_path = f"{config.dir_path}/res_phase2.json"
    with open(res_file_path) as f:
        res = json.load(f)
    return res

def get_res_new(config):
    config.dir_path = get_save_dir(config)
    # print(config.dir_path)
    res_file_path = f"{config.dir_path}/res_phase2.json"
    with open(res_file_path) as f:
        res = json.load(f)
    return res

def get_res_old(config):
    config.dir_path = get_save_dir(config)
    # print(config.dir_path)
    removed_dims = [str(i) for i in config.removed_dims]
    dir_path = f"results/{config.runtimes} teacher_distance {config.teacher_distance} Hierarchical_threshold_k {config.Hierarchical_threshold_k} recon_mask_type {config.recon_mask_type} random_mask_rate {config.random_mask_rate} pseudo_label_type {config.pseudo_label_type} threshold_k {config.threshold_k} {config.dataset}"+"_".join(removed_dims)
    res_file_path = f"{dir_path}/res_phase2.json"
    with open(res_file_path) as f:
        res = json.load(f)
    return res

def get_base_res(config):
    config.removed_dims = [] # baseline
    config.dir_path = config.dir_path = get_save_dir(config)
    res_file_path = f"{config.dir_path}/res_phase1.json"
    with open(res_file_path) as f:
        res = json.load(f)
    return res

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--hiden_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='MSL')

    parser.add_argument('--mode', type=str, default='train_phase2_only', choices=['train', 'test', 'train_phase2', 'train_phase2_only'])
    parser.add_argument('--data_path', type=str, default='./dataset/SMD')

    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=0.5)
    parser.add_argument('--random_mask_rate', type=float, default=0) # md 效果拔群
    parser.add_argument('--weight_scope', type=float, default=0) 
    parser.add_argument('--pseudo_label', type=str2bool, default=False) 
    parser.add_argument('--min_normal_but_not_max_abnormal', type=bool, default=False)
    parser.add_argument('--re_train', type=str2bool, default=True)
    parser.add_argument('--fine_turn', type=str2bool, default=True)
    parser.add_argument('--fine_turn_lr', type=float, default=1e-4)
    parser.add_argument('--runtimes', type=int, default=0)
    config = parser.parse_args()

    return config

def get_config_new():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--hiden_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='SMD')
    # parser.add_argument('--dataset', type=str, default='WT WT23')
    # parser.add_argument('--dataset', type=str, default='SWAT A4_A5')
    # parser.add_argument('--dataset', type=str, default='SWAT A1_A2')
    # parser.add_argument('--dataset', type=str, default='SMAP')
    # parser.add_argument('--dataset', type=str, default='MSL')
    # parser.add_argument('--dataset', type=str, default='PSM')
    parser.add_argument('--mode', type=str, default='train_phase2_only', choices=['train', 'test', 'train_phase2', 'train_phase2_only'])
    parser.add_argument('--data_path', type=str, default='./dataset/SMD')
    # parser.add_argument('--data_path', type=str, default='./dataset/SMAP')
    # parser.add_argument('--data_path', type=str, default='./dataset/MSL')
    # parser.add_argument('--data_path', type=str, default='./dataset/PSM')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=0.5)
    
    # mask
    parser.add_argument('--recon_mask_type', type=str, default='recon_masked', choices=['recon_unmasked', 'recon_masked', 'recon_all', 'none'])
    parser.add_argument('--random_mask_rate', type=float, default=0) # md 效果拔群
    
    # pseudo
    parser.add_argument('--pseudo_label_type', type=str, default='none', choices=['min_normal', 'min_abnormal', 'max_abnormal', 'min_normal_max_abnormal', 'min_normal_double_min_abnormal','soft_min_abnormal','threshold_k_min_abnormal','none'])
    parser.add_argument('--threshold_k', type=float, default=0.5)
    # 未启用
    parser.add_argument('--weight_scope', type=float, default=0) 
    
    # control
    parser.add_argument('--re_train', type=str2bool, default=True)
    parser.add_argument('--fine_turn', type=str2bool, default=True)
    parser.add_argument('--fine_turn_lr', type=float, default=1e-4)
    parser.add_argument('--runtimes', type=int, default=0)
    config = parser.parse_args()
    config.input_c = get_dim_num(config.dataset)
    config.output_c = config.input_c
    if config.recon_mask_type == "none":
        config.random_mask_rate = 0
    
    # baseline run onece 
    config.removed_dims = [] # baseline
    removed_dims = [str(i) for i in config.removed_dims]
    config.dir_path = f"results/{config.runtimes} recon_mask_type {config.recon_mask_type} random_mask_rate {config.random_mask_rate} pseudo_label_type {config.pseudo_label_type} threshold_k {config.threshold_k} {config.dataset}"+"_".join(removed_dims)

    return config

def plot_res(data,xticks,fig_name):
    x = np.arange(len(data))
    # x1 = np.arange(1,20)
    # x2 = np.arange(20,91,10)
    # x = np.concatenate([x1,x2])
    
    plt.cla()
    plt.plot(x,data,label = fig_name)
    plt.xticks(x,xticks)
    plt.legend()
    plt.grid()
    plt.savefig(f"analysis/{fig_name}.pdf")

def ablation_for_mask(k1,k2):
    config = get_argparse()
    
    temp2 = []
    for config.recon_mask_type in recon_mask_type:
        if config.recon_mask_type in ["none"]:continue
        temp1 = [] # 收集数据
        for config.random_mask_rate in random_mask_rate:
            temp = get_data_for_datasets_and_runtimes(config,k1,k2)
            temp1.append(temp)
        temp2.append(temp1)

    res = np.array(temp2)
    res_for_plot = res.mean(axis=-1)[:,:,:,0]
    res_for_plot2 = res_for_plot.mean(axis=-1)
    res_for_plot3 = res_for_plot.max(axis=-2)
    res_mean = res.mean(axis=-1).reshape(-1,3*len(datasets))
    # res_mean = res.max(axis=-1).reshape(-1,3*len(datasets))
    
    res_mean = np.around(res_mean, decimals=4)
    
    recon_mask_type_2 = []
    for i in recon_mask_type:
        if i in ["none"]:continue
        recon_mask_type_2.append(i)
    
    random_mask_rate_index_name = [f"{i}_{j}" for i in recon_mask_type_2 for j in random_mask_rate]
    res_pd = pd.DataFrame(res_mean, index=random_mask_rate_index_name)
    res_pd.to_csv("ablation_for_mask.csv")

    random_mask_rate_index_name = {}
    cnt = 0
    for i in recon_mask_type_2:
        temp = []
        for j in random_mask_rate:
            temp.append(f"{j}")
        plot_res(res_for_plot2[cnt],temp,f"mask_rate_ablation_for_{i}_mean_dataset_aggregation")
        # plot_mask_rate_ablation(res_for_plot3[cnt],temp,f"mask_rate_ablation_for_{i}_max")
        random_mask_rate_index_name[i] = temp
        for j in range(len(datasets)):
            plot_res(res_for_plot[cnt,:,j],temp,f"mask_rate_ablation_for_{i}_{datasets[j]}")
        cnt +=1
    cnt = 0    
    for i in recon_mask_type_2:
        plot_res(res_for_plot3[cnt],datasets,f"mask_rate_ablation_for_{i}_max")
        print(f"{i} max_rate_aggregation f1: {res_for_plot3[cnt].mean(axis=-1)}")
        cnt += 1
    
    # print(res_mean)
    
    # print(f"f1 mean: {res_mean[:,0]} ")
    # print(f"rc mean: {res_mean[:,1]}")  
    # print(f"pc mean: {res_mean[:,2]}\n")
    
    # print(f"f1 base mean: {res_mean[:,3]} ")
    # print(f"rc base mean: {res_mean[:,4]}")
    # print(f"pc base mean: {res_mean[:,5]}")

def get_data_for_datasets_and_runtimes(config,k1,k2):
    temp2 = []
    for config.dataset in datasets:
        temp1 = []
        for config.runtimes in range(k1,k2): 
            res = get_res_new(config)
            # res = get_res_old(config)
            temp =[]
            temp.append(res["f1"])
            temp.append(res["rc"])
            temp.append(res["pc"])
            temp1.append(temp)
        temp2.append(temp1)
    return np.array(temp2).transpose(0,2,1)

def ablation_for_pseudo_label(k1,k2):
    config = get_argparse()

    temp2 = [] # 收集数据
    for config.pseudo_label_type in pseudo_label_type:
        if config.pseudo_label_type in ["Hierarchical_pseudo_label"]: continue
        temp1 = []
        for config.teacher_distance in teacher_distance:
            temp = get_data_for_datasets_and_runtimes(config,k1,k2)
            temp1.append(temp)
        temp2.append(temp1)
            
    res = np.array(temp2)
    # res = res.transpose(0,1,3,4,3)
    res_mean = res.mean(axis=-1).reshape(-1,3*res.shape[-3])
    # res_mean = res.max(axis=-1).reshape(-1,3*len(datasets))
    res_mean = np.around(res_mean, decimals=4)
    
    pseudo_label_type2 = []
    for i in pseudo_label_type:
        if i in ["Hierarchical_pseudo_label"]:continue
        pseudo_label_type2.append(i)
    random_mask_rate_index_name = [f"{i}_{j}" for i in pseudo_label_type2 for j in teacher_distance]
    res_pd = pd.DataFrame(res_mean, index=random_mask_rate_index_name)
    res_pd.to_csv("ablation_for_pseudo_label.csv")
    
def ablation_for_Hierarchical_pseudo_label(k1,k2):
    config = get_argparse()
    config.pseudo_label_type = "Hierarchical_pseudo_label"

    temp2 = [] # 收集数据
    Hierarchical_threshold_k = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    for config.Hierarchical_threshold_k in Hierarchical_threshold_k:
        temp1 = []
        for config.teacher_distance in teacher_distance:
            if config.teacher_distance  in ["kl_distance"]:continue
            temp = get_data_for_datasets_and_runtimes(config,k1,k2)
            temp1.append(temp)
        temp2.append(temp1)
            
    res = np.array(temp2)
    res_mean = res.max(axis=-1).reshape(-1,3*len(datasets))
    res_mean = np.around(res_mean, decimals=4)
    
    teacher_distance_2 = ['recon_distance']
    
    random_mask_rate_index_name = [f"{i}_{j}" for i in Hierarchical_threshold_k for j in teacher_distance_2]
    res_pd = pd.DataFrame(res_mean, index=random_mask_rate_index_name)
    res_pd.to_csv("ablation_for_Hierarchical_pseudo_label.csv")

def distance_distribution():
    config = get_argparse()
    file_path = f"{config.baseline_dir_path}/res.json" 
    with open(file_path) as f:
        res = json.load(f)
    kl_distances = res["thresholds"]#[:10]
    recon_distances = res["recon_thresholds"]#[:10]
    
    plt.cla()
    x = np.arange(len(kl_distances))+1
    plt.plot(x,recon_distances,label=f'recon_distances')
    plt.grid()
    plt.legend()
    plt.savefig("recon_distance_distribution.pdf")
    
    plt.cla()
    x = np.arange(len(kl_distances))+1
    plt.plot(x,kl_distances,label=f'kl_distances')
    plt.grid()
    plt.legend()
    plt.savefig("kl_distance_distribution.pdf")

def find_best_performance(k1,k2):
    config = get_argparse()
    config.pseudo_label_type = "Hierarchical_pseudo_label"
    config.recon_mask_type = "recon_all"

    temp2 = []
    for config.Hierarchical_threshold_k in Hierarchical_threshold_k:
        temp1 = []
        for config.random_mask_rate in random_mask_rate:
            temp = get_data_for_datasets_and_runtimes(config,k1,k2)
            temp1.append(temp)
        temp2.append(temp1)

    res = np.array(temp2)
    res_mean = res.max(axis=-1).reshape(-1,3*len(datasets))
    res_mean = np.around(res_mean, decimals=4)
    xtickes = [f"{i}_{j}" for i in Hierarchical_threshold_k for j in random_mask_rate]
    res_pd = pd.DataFrame(res_mean, index=xtickes)
    res_pd.to_csv("best_performance.csv")
    
def parameter_sensitive(file_path):
    res = pd.read_csv(file_path,header=None, skiprows=1)
    res.drop(0, axis=1, inplace=True)
    res = res.to_numpy()
    Hierarchical_threshold_k = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 28.0, 36.0, 44.0, 52.0, 60.0]
    random_mask_rate = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    res = res.reshape(12,9,3,-1)
    f1_res = res[:,:,:,0]
    f1_k = f1_res.mean(-1).mean(-1)
    f1_r = f1_res.mean(-1).mean(0)
    plt.cla()
    plt.plot(Hierarchical_threshold_k,f1_k,label="k")
    plt.grid()
    plt.legend()
    plt.ylabel("F1")
    # plt.ylim(0.93,0.97)
    plt.xlabel("alpha_2")
    plt.savefig("ps_k.pdf")
    
    plt.cla()
    plt.plot(random_mask_rate,f1_r,label="mask rate")
    plt.grid()
    plt.legend()
    plt.ylabel("F1")
    # plt.ylim(0.93,0.97)
    plt.xlabel("sigma")
    plt.savefig("ps_r.pdf")
    print(res.shape)
    print(f1_k.shape)
    print(f1_r.shape)
    pass

def parameter_sensitive_r(file_path2):
    res = pd.read_csv(file_path2,sep="\t",header=None)
    res.drop(0, axis=1, inplace=True)
    res = res.to_numpy()
    random_mask_rate = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    res = res.reshape(9,3,-1)
    f1_res = res[:,:,0]
    f1_r = f1_res.mean(-1)

    plt.cla()
    plt.plot(random_mask_rate,f1_r,label="mask rate")
    plt.grid()
    plt.legend()
    plt.ylabel("F1")
    plt.ylim(0.95,0.97)
    plt.xlabel("sigma")
    plt.savefig("ps_r_2.pdf")
    print(res.shape)
    print(f1_r.shape)
    pass

def parameter_sensitive_k(file_path, file_path2 ):
    res = pd.read_csv(file_path,sep="\t",header=None)
    res.drop(0, axis=1, inplace=True)
    res = res.to_numpy()
    Hierarchical_threshold_k = [0.01, 0.02,0.04, 0.08, 0.16, 0.32, 0.64]
    res = res.reshape(7,3,-1)
    f1_res = res[:,:,0]
    f1_k = f1_res.mean(-1)


    plt.cla()
    plt.plot(Hierarchical_threshold_k,f1_k,label="alpha")
    plt.grid()
    plt.legend()
    plt.ylabel("F1")
    plt.ylim(0.95,0.97)
    
    # plt.savefig("ps_k_2.pdf")
    
    
    res = pd.read_csv(file_path2,sep="\t",header=None)
    res.drop(0, axis=1, inplace=True)
    res = res.to_numpy()
    random_mask_rate = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    res = res.reshape(9,3,-1)
    f1_res = res[:,:,0]
    f1_r = f1_res.mean(-1)

    # plt.cla()
    plt.plot(random_mask_rate,f1_r,label="sigma")
    plt.grid()
    plt.legend()
    plt.ylabel("F1")
    plt.ylim(0.95,0.97)
    plt.xlabel("alpha/sigma")
    # plt.xlabel("sigma")
    plt.savefig("ps_r_2.pdf")
    print(res.shape)
    print(f1_r.shape)
    pass

def clip_data(data):
    return np.array([data[i] for i in range(int(len(data)*0.989), int(len(data)*0.991))])
    # 989,991 for MSL

def distance_distribution_v2():
    config = get_argparse()
    base_dir = f"{config.baseline_dir_path}"
    dims = get_dim_num(config.dataset)
    test_input = np.load(f"{base_dir}/test_input.npy").reshape(-1,dims)
    test_output = np.load(f"{base_dir}/test_output.npy").reshape(-1,dims)
    retrain_test_input = np.load(f"{base_dir}/retrain_test_input.npy")
    retrain_test_output = np.load(f"{base_dir}/retrain_test_output.npy")
    labels = np.load(f"{base_dir}/test_labels.npy")
    
    # test_input, scaler = normalize_data(test_input)
    # test_output, _ = normalize_data(test_output,scaler)
    # retrain_test_input, scaler = normalize_data(retrain_test_input)
    # retrain_test_output, _ = normalize_data(retrain_test_output,scaler)
    
    distance1 =  ((test_output-test_input)**2).mean(-1)
    distance2 =  ((retrain_test_output-retrain_test_input)**2).mean(-1)
    
    index1 = np.argsort(distance1)
    distance1 = distance1[index1]
    index2 = np.argsort(distance2)
    distance2 = distance2[index2]
    
    thresh1 = np.percentile(distance1,99)*0.05
    thresh2 = np.percentile(distance2,99)*0.05
   
    colors1 = []
    colors2 = []
    for i in labels:
        if i:
            colors1.append("r")
            colors2.append("m")
        else:
            colors1.append("g")
            colors2.append("b")
    colors1 = np.array(colors1)[index1]
    colors2 = np.array(colors2)[index2]
    
    colors1_ = clip_data(colors1)
    colors2_ = clip_data(colors2)
    distance1_ = clip_data(distance1)*0.05
    distance2_ = clip_data(distance2)*0.05
    
    x = np.arange(len(distance1_))
    plt.scatter(x,distance1_,marker='o',c = colors1_,label = "Anomaly Transformer", alpha=0.5) # s=2,
    plt.scatter(x,distance2_,marker='^',c= colors2_,label = "Our", alpha=0.5) # s=2,
    plt.plot(x,np.ones_like(x)*thresh1,label = "Decision Boundary", alpha=0.5)
    plt.ylabel("Reconstruction Error")
    plt.xlabel("Timestamp")
    # plt.plot(x,np.ones_like(x)*thresh2,label = "decision boundary2", alpha=0.5)
    plt.legend()
    plt.savefig(f"distance_distribution_v2_{config.dataset}.pdf")
    pass
    

if __name__ == '__main__':
    # distance_distribution()
    # ablation_for_mask(39,41)
    # ablation_for_pseudo_label(31,36)
    # ablation_for_Hierarchical_pseudo_label(31,36)
    # find_best_performance(40,42)
    # parameter_sensitive("best_performance.csv")
    # parameter_sensitive_r("ps_r.csv")
    # parameter_sensitive_k("ps_k.csv","ps_r.csv")
    distance_distribution_v2()
                
        