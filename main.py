import os
import argparse
from torch.backends import cudnn
from utils.utils import *
from solver_v2 import Solver

datasets = ['SMAP','MSL','PSM'] # 'SMD', 'SWAT A4_A5',

# randome_mask
recon_mask_type = ['recon_unmasked', 'recon_masked', 'recon_all', 'none']
random_mask_rate = [0.1,
                    0.2,
                    0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# pseudo_label
pseudo_label_type = ['invers_focal_loss','focal_loss','min_normal', 'max_abnormal', 'min_normal_max_abnormal', 'min_normal_double_min_abnormal', 'min_abnormal', 'Hierarchical_pseudo_label', 'self_softmax', 'soft_min_abnormal'] # 直接试一下reconstruction-based focal loss #然后发现也不行，因为异常检测，不能把异常也重构的太好，所以需要设阈值
teacher_distance = ['recon_distance','kl_distance']
Hierarchical_threshold_k = [1.0,2.0,4.0,8.0,12.0,16.0,20.0,
                            28.0,36.0,44.0,52.0,60.0
                            ]
fine_distance = teacher_distance


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
        solver.test()
    elif config.mode == 'train_phase2':
        solver.train()
        solver.train_phase2()
    elif config.mode == 'train_phase2_only':
        solver.train_phase2()
    elif config.mode == 'test':
        solver.test()

    return solver

def get_dim_num(dataset = "SMD"):
    if dataset == "SMD": return 38
    elif dataset == "SMAP": return 25
    elif dataset == "MSL": return 55
    elif dataset == "PSM": return 25
    elif dataset == "WT": return 10
    elif dataset == "BATADAL": return 10
    else:
        dataset2 = dataset.split(" ")[0]
        group = dataset.split(" ")[1]
        if group == "A1_A2": return 50
        elif group == "A4_A5": return 77
        return get_dim_num(dataset2)

def get_save_dir(config):
    removed_dims = [str(i) for i in config.removed_dims]
    return f"results/{config.runtimes} fine_distance {config.fine_distance} test_distance {config.test_distance} teacher_distance {config.teacher_distance} Hierarchical_threshold_k {config.Hierarchical_threshold_k} recon_mask_type {config.recon_mask_type} random_mask_rate {config.random_mask_rate} pseudo_label_type {config.pseudo_label_type} threshold_k {config.threshold_k} {config.dataset}"+"_".join(removed_dims)
    # fine_distance {config.fine_distance} 
    
def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--hiden_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='MSL')
    parser.add_argument('--mode', type=str, default='train_phase2', choices=['train', 'test', 'train_phase2', 'train_phase2_only'])
    parser.add_argument('--data_path', type=str, default='./dataset/MSL')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=0.1)
    
    # fine trun 
    parser.add_argument('--fine_distance', type=str, default='kl_distance', choices=teacher_distance)
    parser.add_argument('--test_distance', type=str, default='kl_distance', choices=teacher_distance)

    # mask
    parser.add_argument('--recon_mask_type', type=str, default='recon_all', choices=recon_mask_type)
    parser.add_argument('--random_mask_rate', type=float, default=0.2,choices=random_mask_rate) 
    
    # pseudo
    parser.add_argument('--pseudo_label_type', type=str, default='Hierarchical_pseudo_label', choices=pseudo_label_type)
    parser.add_argument('--threshold_k', type=float, default=0.5)
    parser.add_argument('--teacher_distance', type=str, default='recon_distance', choices=teacher_distance)
    parser.add_argument('--Hierarchical_threshold_k', type=float, default=2,choices=Hierarchical_threshold_k)
    # 未启用
    parser.add_argument('--weight_scope', type=float, default=0) 
    
    # control
    parser.add_argument('--re_train', type=str2bool, default=True)
    parser.add_argument('--compare_model_output', type=str2bool, default=True)
    parser.add_argument('--fine_turn', type=str2bool, default=True)
    parser.add_argument('--fine_turn_lr', type=float, default=1e-4)
    parser.add_argument('--runtimes', type=int, default=40)
    config = parser.parse_args()
    config.input_c = get_dim_num(config.dataset)
    config.output_c = config.input_c
    if config.recon_mask_type == "none":
        config.random_mask_rate = 0
    
    # baseline run onece 
    config.removed_dims = [] # baseline
    config.baseline_dir_path = f"results/baseline {config.dataset}"
    config.dir_path = get_save_dir(config)
    if config.recon_mask_type == "none" and config.pseudo_label_type == "none":
        config.dir_path = config.baseline_dir_path
    if not os.path.exists(config.dir_path):
        os.mkdir(config.dir_path)
    if not os.path.exists(config.baseline_dir_path):
        os.mkdir(config.baseline_dir_path)
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    return config

if __name__ == '__main__':
    config = get_argparse()
    main(config)
