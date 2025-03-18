import numpy as np
import random
import torch
import argparse
import time
import pickle

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from utils.train import *
from utils.loader import *
from utils.utility import *
from utils.model import *
from datetime import datetime

import wandb

### Make argument parser(hyper-parameters)
def get_args():
    parser = argparse.ArgumentParser()
    ### WanDB
    parser.add_argument('--run_name', default='ppmi', help='Name of wandb run')
    ### Data
    parser.add_argument('--data', default='ppmi', help='Type of dataset')
    parser.add_argument('--nclass', default=5, help='Number of classes')
    parser.add_argument('--adjacency_path', default='/path_to_adj', help='set path to adjacency matrices')
    parser.add_argument('--save_dir', default='./logs/', help='directory for saving weight file')

    ### Condition
    parser.add_argument('--t_init_min', type=float, default=-2.0, help='Init value of t')
    parser.add_argument('--t_init_max', type=float, default=2.0, help='Init value of t')
    parser.add_argument('--seed_num', type=int, default=100, help='Number of random seed')
    parser.add_argument('--device', type=int, default=0, help='Which gpu to use')
    parser.add_argument('--model', type=str, default='agt', help='Models to use') 
    parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')

    ### Experiment
    parser.add_argument('--beta', type=float, default=0.005, help='weight of temporal regularization, alpha in the paper') 
    parser.add_argument('--split_num', type=int, default=5, help="Number of splits for k-fold")
    parser.add_argument('--batch_size', type=int, default=512, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--hidden_units', type=int, default=8, help='Number of hidden units')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learing rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer adam/sgd')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 regularization') # 0.0005

    ### Parameters for training GAT 
    parser.add_argument('--num_head_attentions', type=int, default=16, help='Number of head attentions')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for the leaky relu')
    args = parser.parse_args()

    ### Parameters for training Exact 
    parser.add_argument('--use_t_local', type=int, default=1, help='Whether t is local or global (0:global / 1:local)')
    parser.add_argument('--t_lr', type=float, default=1, help='t learning rate')
    parser.add_argument('--t_loss_threshold', type=float, default=0.01, help='t loss threshold')
    parser.add_argument('--t_lambda', type=float, default=1, help='t lambda of loss function')
    parser.add_argument('--t_threshold', type=float, default=0.1, help='t threshold')
    
    return args

### Control the randomness of all experiments
def set_randomness(seed_num):
    torch.manual_seed(seed_num) # Pytorch randomness
    np.random.seed(seed_num) # Numpy randomness
    random.seed(seed_num) # Python randomness
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num) # Current GPU randomness
        torch.cuda.manual_seed_all(seed_num) # Multi GPU randomness

### Main function
def main():
    args = get_args()
    set_randomness(args.seed_num)
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

    wandb.init(project="graph classification")
    wandb.run.name = args.run_name
    wandb.run.save()
    
    if args.data == 'adni_ct' or args.data == 'adni_fdg':
        ### Load fully-preprocessed data
        if args.data == 'adni_ct':
            args.data_path = 'data_path_to_ct'
            used_features = ['cortical thickness']
        elif args.data == 'adni_fdg':
            args.data_path = 'data_path_to_fdg'
            used_features = ['FDG SUVR']

        A, X, y, eigenvalues, eigenvectors = load_saved_data(args)
        args.nclass = 5

    elif args.data == 'ppmi':
        with open('data_path_to_ppmi', 'rb') as fr:
            data = pickle.load(fr)
            A = data[0]
            X = data[1]
            y = data[2]
            eigenvectors = data[3]
            eigenvalues = data[4]
        args.nclass = 3
        
    ### K-fold cross validation
    stratified_train_test_split = StratifiedKFold(n_splits=args.split_num)

    avl, ava, avac, avpr, avsp, avse, avf1s, ts = list([] for _ in range(8))
    average_acc_per_fold = []
    average_sens_per_fold, average_prec_per_fold = [], []

    idx_pairs = []

    for train_idx, test_idx in stratified_train_test_split.split(A, y):
        idx_train = torch.LongTensor(train_idx)
        idx_test = torch.LongTensor(test_idx)
        idx_pairs.append((idx_train, idx_test))

    ### Utilize GPUs for computation
    if torch.cuda.is_available() and args.model != 'svm':
        A = A.to(device) # Shape: (# subjects, # ROI feature, # ROI X)
        X = X.to(device) # Shape: (# subjects, # ROI X, # used X)
        y = y.to(device) # Shape: (# subjects)
    
        eigenvalues = eigenvalues.to(device) # Shape: (# subjects, # ROI feature)
        eigenvectors = eigenvectors.to(device) # Shape: (# subject, # ROI_feature, # ROI_feature)
        # laplacians = laplacians.to(device)
    
    if args.model == 'agt':
        num_ROI_features = X.shape[1]
        num_used_features = X.shape[2] * 2 # stack[x, xs]
    elif args.model == 'svm':
        num_ROI_features = None
        num_used_features = None
    else:
        num_ROI_features = X.shape[1]
        num_used_features = X.shape[2] 


    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir_ = os.path.join(args.save_dir, current_time)
    args.save_dir = save_dir_
    os.makedirs(save_dir_, exist_ok=False)
    wandb.config.update(args)
    print('save directory: ', args.save_dir )


    for i, idx_pair in enumerate(idx_pairs):
        print("\n")
        print(f"=============================== Fold {i+1} ===============================")
        
        ### Build data loader
        data_loader_train, data_loader_test = build_data_loader(args, idx_pair, A, X, y, eigenvalues, eigenvectors)

        ### Select the model to use
        model = select_model(args, num_ROI_features, num_used_features, A, y)

        optimizer = select_optimizer(args, model)        
        if args.model == 'agt':
            if args.data == 'adni_ct':
                data_pretrain_path = 'trained_exact/adni_ct'
                if i == 0:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_0_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_0_t.pt'))
                elif i == 1:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_1_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_1_t.pt'))
                elif i == 2:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_2_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_2_t.pt'))
                elif i == 3:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_3_model.pt')) 
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_3_t.pt')) 
                elif i == 4:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_4_model.pt'))  
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_4_t.pt'))
            
            elif args.data == 'ppmi':
                data_pretrain_path = 'trained_exact/ppmi'
                if i == 0:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_0_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_0_t.pt'))
                elif i == 1:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_1_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_1_t.pt'))
                elif i == 2:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_2_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_2_t.pt'))
                elif i == 3:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_3_model.pt')) 
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_3_t.pt')) 
                elif i == 4:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_4_model.pt'))  
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_4_t.pt'))

            elif args.data == 'adni_fdg':
                data_pretrain_path = 'trained_exact/adni_fdg'
                if i == 0:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_0_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_0_t.pt'))
                elif i == 1:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_1_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_1_t.pt'))
                elif i == 2:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_2_model.pt'))
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_2_t.pt'))
                elif i == 3:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_3_model.pt')) 
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_3_t.pt')) 
                elif i == 4:
                    p_net = torch.load(os.path.join(data_pretrain_path, 'fold_4_model.pt'))  
                    p_t = torch.load(os.path.join(data_pretrain_path, 'fold_4_t.pt'))

            trainer = select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, A, cv_idx = i, pretrained_net=p_net, pretrained_t=p_t) 

        elif args.model == 'exact':
            trainer = select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, A, cv_idx=i) 
        else: 
            trainer = select_trainer(args, device, model, optimizer, data_loader_train, data_loader_test, A)
            
        ### Train and test
        val_acc_list, val_sens_list, val_prec_list = trainer.train()
        
        average_acc_per_fold.append(val_acc_list)
        average_sens_per_fold.append(val_sens_list)
        average_prec_per_fold.append(val_prec_list)

        ### best model 
        model_path = os.path.join(args.save_dir, '{}.pth'.format(i)) # get model from logs/datetime/
        test_model = select_model(args, num_ROI_features, num_used_features, A, y)
        losses, accuracies, cf_accuracies, cf_precisions, cf_specificities, cf_sensitivities, cf_f1score, t = trainer.load_and_test(test_model, model_path)

        avl.append(losses)
        ava.append(accuracies)
        avac.append(cf_accuracies)
        avpr.append(cf_precisions)
        avsp.append(cf_specificities)
        avse.append(cf_sensitivities)
        avf1s.append(cf_f1score)
        ts.append(t)
        wandb.config.update({"fold_" + str(i) + '_acc': round(accuracies, 4)})

    class_info = y.tolist()
    cnt = Counter(class_info)

    wandb.config.update({"avg_acc": round(np.mean(ava), 4),
    "avg_prec": round(np.mean(avpr), 4),
    "avg_sens": round(np.mean(avse), 4),
    "std_acc": round(np.std(ava), 4),
    "std_prec": round(np.std(avpr), 4),
    "std_sens": round(np.std(avse), 4),
    })

    print("len(average_acc_per_fold): ", len(average_acc_per_fold))
    print("len(average_acc_per_fold[0]): ", len(average_acc_per_fold[0]))

    average_acc_per_epoch = [list(x) for x in zip(*average_acc_per_fold)]
    average_sens_per_epoch = [list(x) for x in zip(*average_sens_per_fold)]
    average_prec_per_epoch = [list(x) for x in zip(*average_prec_per_fold)]
    
    print("len(average_acc_per_epoch): ", len(average_acc_per_epoch))
    print("len(average_acc_per_epoch[0]): ", len(average_acc_per_epoch[0]))
    best_val_acc = -1
    best_idx = -1
    for idx in range(args.epochs):
        avg_val_acc = np.mean(average_acc_per_epoch[idx])
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_idx = idx
            
            best_val_sens = np.mean(average_sens_per_epoch[idx])
            best_val_prec = np.mean(average_prec_per_epoch[idx])

            acc_std = np.std(average_acc_per_epoch[idx])
            sens_std = np.std(average_sens_per_epoch[idx])
            prec_std = np.std(average_prec_per_epoch[idx])

    print('best idx: ', best_idx)
    print('best val acc: ', best_val_acc)

    print("all accuracies: ", average_acc_per_epoch[best_idx])

    wandb.config.update({"best_val_acc": best_val_acc})
    wandb.config.update({"best_val_sens": best_val_sens})
    wandb.config.update({"best_val_prec": best_val_prec})

    wandb.config.update({"best_acc_std": acc_std})
    wandb.config.update({"best_sens_std": sens_std})
    wandb.config.update({"best_prec_std": prec_std})

    wandb.config.update({"best_idx": best_idx})

    # for i in range(5):
    #     wandb.config.update({"fold_" + str(i) + '_acc': round(ava[i], 4)})

    ### Show results
    print("--------------- Result ---------------")
    if args.data == 'adni':
        print(f"Used X:        {used_features}")
    print(f"Label distribution:   {cnt}")
    print(f"{args.split_num}-Fold test loss:     {avl}")
    print(f"{args.split_num}-Fold test accuracy: {ava}")
    print("---------- Confusion Matrix ----------")
    #print(f"{args.split_num}-Fold accuracy:      {avac}")
    print(f"{args.split_num}-Fold precision:     {avpr}")
    print(f"{args.split_num}-Fold specificity:   {avsp}")
    print(f"{args.split_num}=Fold sensitivity:   {avse}")
    #print(f"{args.split_num}=Fold f1 score:      {avf1s}")
    print("-------------- Mean, Std --------------")
    print(f"Mean:  {np.mean(ava):.4f} {np.mean(avpr):.4f} {np.mean(avse):.4f} {np.mean(avsp):.4f}")
    print(f"Std:   {np.std(ava):.4f}  {np.std(avpr):.4f}  {np.std(avse):.4f} {np.std(avsp):.4f}")


if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    process_time = time.time() - start_time
    hour = int(process_time // 3600)
    minute = int((process_time - hour * 3600) // 60)
    second = int(process_time % 60)
    print(f"\nTime: {hour}:{minute}:{second}")