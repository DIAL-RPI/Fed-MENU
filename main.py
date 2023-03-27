import sys
import os
import numpy as np
import random
import time
from itertools import cycle
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data
import math
from dataset import create_folds, ClientDataset
from model import MENUNet
from loss import marginal_loss, exclusion_loss, loc_dice_and_ce_loss, dice_and_ce_loss
from utils import resample_array, output2file, generate_gauss_weight
from metric import eval
from config import cfg

def initial_net(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def get_best_model_name(dir):
    for fn in os.listdir(dir):
        if fn.startswith('cp_commu_') and fn.endswith('.pth.tar'):
            return fn
    return ''

def initialization():
    global_model = nn.DataParallel(module=MENUNet(in_ch=1, base_ch=20, cls_num=cfg['cls_num']))
    global_model.cuda()
    initial_net(global_model)
    
    nodes = []
    weight_sum = 0
    node_cls_flag = []
    for node_id, [node_name, d_name, d_path, fraction] in enumerate(cfg['node_list']):

        folds, _ = create_folds(d_name, d_path, node_name, fraction, exclude_case=cfg['exclude_case'])

        # create training fold
        train_fold = folds[0]
        d_train = ClientDataset(train_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=True, full_sampling=False, enforce_fg=False, fixed_sample=False)
        dl_train = data.DataLoader(dataset=d_train, batch_size=cfg['batch_size'], shuffle=True, pin_memory=True, drop_last=True, num_workers=cfg['cpu_thread'])

        # create validaion fold
        val_fold = folds[1]
        d_val = ClientDataset(val_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=False, full_sampling=False, enforce_fg=True, fixed_sample=True)
        dl_val = data.DataLoader(dataset=d_val, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])

        # create testing fold
        test_fold = folds[2]
        d_test = ClientDataset(test_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=False, full_sampling=True, enforce_fg=False, fixed_sample=False)
        dl_test = data.DataLoader(dataset=d_test, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])

        print('{0:s}: train/val/test = {1:d}/{2:d}/{3:d}'.format(node_name, len(d_train), len(d_val), len(d_test)))
        weight_sum += len(d_train)

        local_model = nn.DataParallel(module=MENUNet(in_ch=1, base_ch=20, cls_num=cfg['cls_num']))
        local_model.cuda()
        local_model.load_state_dict(global_model.state_dict())

        node_enabled_encoders = []
        for c in range(cfg['cls_num']):
            if (c+1) in cfg['label_map'][d_name[0]].values():
                node_enabled_encoders.append(c)

        optimizer1 = optim.SGD(local_model.module.get_s1_parameters(node_enabled_encoders), lr=cfg['lr'], momentum=0.99, nesterov=True, weight_decay=3e-5)
        optimizer2 = optim.SGD(local_model.module.get_s2_parameters(node_enabled_encoders), lr=cfg['lr'], momentum=0.99, nesterov=True, weight_decay=3e-5)
        
        lambda_func = lambda epoch: (1 - epoch / (cfg['commu_times'] * cfg['epoch_per_commu']))**0.9
        scheduler1 = optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda=lambda_func)
        scheduler2 = optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda=lambda_func)
        
        nodes.append([local_model, [optimizer1, optimizer2], [scheduler1, scheduler2], node_name, len(d_train), dl_train, dl_val, dl_test])
        
        cls_labels = []
        for dn in d_name:
            cls_labels.extend(cfg['label_map'][dn].values())
        tmp_flag = []
        for c in range(cfg['cls_num']):
            if (c+1) in cls_labels:
                tmp_flag.append(True)
            else:
                tmp_flag.append(False)
        node_cls_flag.append(tmp_flag)

    for i in range(len(nodes)):
        nodes[i][4] = nodes[i][4] / weight_sum
        print('Weight of {0:s}: {1:f}'.format(nodes[i][3], nodes[i][4]))

    return global_model, nodes, node_cls_flag

def initialize_ood_node():
    ood_nodes = []
    for _, [node_name, d_name, d_path, fraction] in enumerate(cfg['ood_node_list']):

        folds, _ = create_folds(d_name, d_path, node_name, fraction, exclude_case=cfg['exclude_case'])

        # create out-of-distribution (ood) testing fold
        test_fold = folds[0]
        test_fold.extend(folds[1])
        test_fold.extend(folds[2])
        d_test = ClientDataset(test_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=False, full_sampling=True, enforce_fg=False, fixed_sample=False)
        dl_test = data.DataLoader(dataset=d_test, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])

        print('{0:s}: test = {1:d}'.format(node_name, len(d_test)))
        
        ood_nodes.append([None, None, None, node_name, None, None, None, dl_test])

    return ood_nodes

def train_local_model(local_model, optimizer, scheduler, data_loader, epoch_num):
    train_loss = 0
    train_loss_num = 0
    for epoch_id in range(epoch_num):

        t0 = time.perf_counter()

        epoch_loss = 0
        epoch_loss_num = 0
        batch_id = 0
        for batch in data_loader:
            image = batch['data'].cuda()
            label = batch['label'].cuda()

            N = len(image)

            lmap = cfg['label_map'][batch['dataset'][0]]
            class_flag = [1]
            node_enabled_encoders = []
            for c in range(cfg['cls_num']):
                if (c+1) in lmap.values():
                    class_flag.append(1)
                    node_enabled_encoders.append(c)
                else:
                    class_flag.append(0)

            # Step 1: optimize whole network using marginal/exclusion loss
            optimizer[0].zero_grad()
            pred = local_model(image)

            print_line = 'Epoch {0:d}/{1:d} (train) --- Progress {2:5.2f}% (+{3:02d})'.format(
                epoch_id+1, epoch_num, 100.0 * batch_id * cfg['batch_size'] / len(data_loader.dataset), N)
            
            l_mce, l_mdice = marginal_loss(pred[0], label, class_flag)
            l_ece, l_edice = exclusion_loss(pred[0], label, class_flag)
            loss_sup = l_mdice + l_mce + l_edice + l_ece

            epoch_loss += loss_sup.item()

            print_line += ' --- S1: {0:.6f}'.format(loss_sup.item())
            
            loss_sup.backward()
            optimizer[0].step()

            del pred, loss_sup

            # Step 2: optimize sub-networks of labeled ROIs using CE/Dice loss
            optimizer[1].zero_grad()
            pred_2 = local_model(image, node_enabled_encoders)
            loss_sup = 0
            for encoder_id, [pred_s1, pred_s2, pred_s3, pred_s4] in zip(node_enabled_encoders, pred_2):             
                l_ce1, l_dice1 = loc_dice_and_ce_loss(pred_s1, label, encoder_id+1)
                l_ce2, l_dice2 = loc_dice_and_ce_loss(pred_s2, label, encoder_id+1)
                l_ce3, l_dice3 = loc_dice_and_ce_loss(pred_s3, label, encoder_id+1)
                l_ce4, l_dice4 = loc_dice_and_ce_loss(pred_s4, label, encoder_id+1)
                loss_sup += l_dice1 + l_ce1 + l_dice2 + l_ce2 + l_dice3 + l_ce3 + l_dice4 + l_ce4
            epoch_loss += loss_sup.item()
            print_line += ' --- S2: {0:.6f}'.format(loss_sup.item())
            
            loss_sup.backward()
            optimizer[1].step()

            del image, label, pred_2, loss_sup
            batch_id += 1
            epoch_loss_num += 1
            print(print_line)

        train_loss += epoch_loss
        train_loss_num += epoch_loss_num
        epoch_loss = epoch_loss / epoch_loss_num
        lr = scheduler[0].get_last_lr()[0]

        print_line = 'Epoch {0:d}/{1:d} (train) --- Loss: {2:.6f} --- Lr: {3:.6f}'.format(epoch_id+1, epoch_num, epoch_loss, lr)
        print(print_line)

        scheduler[0].step()
        scheduler[1].step()

        t1 = time.perf_counter()
        epoch_t = t1 - t0
        print("Epoch time cost: {h:>02d}:{m:>02d}:{s:>02d}\n".format(
            h=int(epoch_t) // 3600, m=(int(epoch_t) % 3600) // 60, s=int(epoch_t) % 60))

    train_loss = train_loss / train_loss_num

    return train_loss

def communication(global_model, nodes, node_cls_flag):

    # for each sub-encoder, we calculate the fusion weight according to the labeled client datasets
    sub_encoder_weights = []
    for c in range(cfg['cls_num']):
        w = []
        w_sum = 0
        for node_id in range(len(nodes)):
            if node_cls_flag[node_id][c] > 0:
                w.append(1.0)
                w_sum += 1.0
            else:
                w.append(0.5)
                w_sum += 0.5

        for node_id in range(len(nodes)):
            w[node_id] = w[node_id] / w_sum
        sub_encoder_weights.append(w)

    for key in global_model.state_dict().keys():
        temp = torch.zeros_like(global_model.state_dict()[key])
        if '.sub_encoders.' in key: # average fusion weighted by the weights calculated above
            for c in range(cfg['cls_num']):
                if '.sub_encoders.{0:d}.'.format(c) in key:
                    for node_id in range(len(nodes)):
                        temp += sub_encoder_weights[c][node_id] * nodes[node_id][0].state_dict()[key]
                    break
        else: # average fusion weighted by the size of client datasets
            for node_id in range(len(nodes)):
                temp += nodes[node_id][4] * nodes[node_id][0].state_dict()[key]
        global_model.state_dict()[key].data.copy_(temp)
        for node_id in range(len(nodes)):
            nodes[node_id][0].state_dict()[key].data.copy_(global_model.state_dict()[key])
    return global_model

# mode: 'val' or 'test'
# commu_iters: communication iteration index, only used when mode == 'val' (for command line output)
def eval_global_model(global_model, nodes, result_path, mode, commu_iters):
    t0 = time.perf_counter()
    global_model.eval()
    gt_entries = []
    if mode == 'val':
        val_dice = 0.0
        val_dice_per_cls = np.zeros(cfg['cls_num'], dtype=float)
        val_num_per_cls = np.zeros(cfg['cls_num'], dtype=np.uint8)
        val_dice_num = 0

    for node_id, [local_model, optimizer, scheduler, node_name, node_weight, dl_train, dl_val, dl_test] in enumerate(nodes):
        if mode == 'val':
            data_loader = dl_val
            metric_fname = 'metric_validation-{0:04d}'.format(commu_iters+1)
            
            print('Validation ({0:d}/{1:d}) on Node #{2:d}'.format(commu_iters+1, cfg['commu_times'], node_id))
        elif mode == 'test':
            data_loader = dl_test
            metric_fname = 'metric_testing'
            
            print('Testing on Node #{0:d}'.format(node_id))
        elif mode == 'test-ood': # test on OoD (out-of-distribution) data
            data_loader = dl_test
            metric_fname = 'metric_testing_ood'
            
            print('Testing on OoD Node #{0:d}'.format(node_id))

        whole_prob_buffer = None
        whole_prob_weight = None
        for batch_id, batch in enumerate(data_loader):
            image = batch['data']
            N = len(image)

            image = image.cuda()

            prob = global_model(image)
            mask = torch.argmax(prob, dim=1, keepdim=True).detach().cpu().numpy().copy().astype(dtype=np.uint8)
            mask = np.squeeze(mask, axis=1)

            print_line = '{0:s} --- Progress {1:5.2f}% (+{2:d})'.format(
                mode, 100.0 * batch_id * cfg['test_batch_size'] / len(data_loader.dataset), N)
            print(print_line)

            if mode == 'val':
                label = batch['label'].cuda()
                for i in range(N):
                    d_name = batch['dataset'][i]
                    lmap = cfg['label_map'][d_name]
                    class_flag = [1]
                    for c in range(cfg['cls_num']):
                        if (c+1) in lmap.values():
                            class_flag.append(1)
                        else:
                            class_flag.append(0)
                    _, l_dice, l_per_cls, n_per_cls = dice_and_ce_loss(prob[i:i+1,:], label[i:i+1,:], class_flag)
                    val_dice += 1.0 - l_dice.item()
                    val_dice_per_cls += l_per_cls
                    val_num_per_cls += n_per_cls
                    del l_dice, l_per_cls, n_per_cls
                val_dice_num += N
                del label
            else:
                prob = prob.detach().cpu().numpy().copy()
                for i in range(N):
                    stack_size = batch['size'][i].numpy()
                    stack_weight = generate_gauss_weight(stack_size[2], stack_size[1], stack_size[0])
                    if whole_prob_buffer is None:
                        whole_mask_size = stack_size.copy()
                        whole_mask_size[0] = (batch['patch_grid_size'][i][0]-1)*batch['patch_stride'][i][0]+stack_size[0]
                        whole_mask_size[1] = (batch['patch_grid_size'][i][1]-1)*batch['patch_stride'][i][1]+stack_size[1]
                        whole_mask_size[2] = (batch['patch_grid_size'][i][2]-1)*batch['patch_stride'][i][2]+stack_size[2]
                        whole_mask_origin = batch['origin'][i].numpy()
                        whole_mask_spacing = batch['spacing'][i].numpy()
                        whole_prob_buffer = np.zeros((prob.shape[1], whole_mask_size[2], whole_mask_size[1], whole_mask_size[0]), dtype=prob.dtype)
                        whole_prob_weight = np.zeros((prob.shape[1], whole_mask_size[2], whole_mask_size[1], whole_mask_size[0]), dtype=prob.dtype)
                        
                    stack_start_pos = [0, 0, 0]
                    stack_end_pos = [0, 0, 0]
                    stack_start_pos[0] = batch['patch_pos'][i][0] * batch['patch_stride'][i][0]
                    stack_start_pos[1] = batch['patch_pos'][i][1] * batch['patch_stride'][i][1]
                    stack_start_pos[2] = batch['patch_pos'][i][2] * batch['patch_stride'][i][2]
                    stack_end_pos[0] = stack_start_pos[0] + stack_size[0]
                    stack_end_pos[1] = stack_start_pos[1] + stack_size[1]
                    stack_end_pos[2] = stack_start_pos[2] + stack_size[2]
                    whole_prob_buffer[:, stack_start_pos[2]:stack_end_pos[2], stack_start_pos[1]:stack_end_pos[1], stack_start_pos[0]:stack_end_pos[0]] += prob[i,:] * stack_weight
                    whole_prob_weight[:, stack_start_pos[2]:stack_end_pos[2], stack_start_pos[1]:stack_end_pos[1], stack_start_pos[0]:stack_end_pos[0]] += stack_weight
                    if batch['eof'][i] == True:
                        whole_prob_buffer = whole_prob_buffer / whole_prob_weight
                        resampled_prob = np.zeros((whole_prob_buffer.shape[0], batch['org_size'][i][2], batch['org_size'][i][1], batch['org_size'][i][0]), dtype=prob.dtype)
                        for c in range(whole_prob_buffer.shape[0]):
                            resampled_prob[c,:] = resample_array(whole_prob_buffer[c,:], whole_mask_size, whole_mask_spacing, whole_mask_origin, 
                                    batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy(), linear=True)
                        whole_mask = resampled_prob.argmax(axis=0).astype(np.uint8)
                        output2file(whole_mask, batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy(), 
                                '{}/{}@{}.nii.gz'.format(result_path, batch['dataset'][i], batch['case'][i]))
                        whole_prob_buffer = None
                        whole_prob_weight = None
                        gt_entries.append([batch['dataset'][i], batch['case'][i], batch['label_fname'][i]])

            del image, prob
        
    if mode == 'val':
        val_dice_per_cls = val_dice_per_cls / val_num_per_cls
        seg_dsc_m = val_dice_per_cls.mean()
        seg_dsc = None

        print_line = 'Validation result (iter = {0:d}/{1:d}) --- DSC {2:.2f}% ({3:s})'.format(
            commu_iters+1, cfg['commu_times'], seg_dsc_m*100.0, '/'.join(['%.2f']*len(val_dice_per_cls)) % tuple(val_dice_per_cls*100.0))
    else:        
        seg_dsc, seg_asd, seg_hd, seg_dsc_m, seg_asd_m, seg_hd_m = eval(
            pd_path=result_path, gt_entries=gt_entries, label_map=cfg['label_map'], cls_num=cfg['cls_num'], 
            metric_fn=metric_fname, calc_asd=(mode != 'val'))

        print_line = 'Testing results --- DSC {0:.2f} ({1:s})% --- ASD {2:.2f} ({3:s})mm --- HD {4:.2f} ({5:s})mm'.format(
            seg_dsc_m*100.0, '/'.join(['%.2f']*len(seg_dsc[:,0])) % tuple(seg_dsc[:,0]*100.0), 
            seg_asd_m, '/'.join(['%.2f']*len(seg_asd[:,0])) % tuple(seg_asd[:,0]),
            seg_hd_m, '/'.join(['%.2f']*len(seg_hd[:,0])) % tuple(seg_hd[:,0]))
    print(print_line)
    t1 = time.perf_counter()
    eval_t = t1 - t0
    print("Evaluation time cost: {h:>02d}:{m:>02d}:{s:>02d}\n".format(
        h=int(eval_t) // 3600, m=(int(eval_t) % 3600) // 60, s=int(eval_t) % 60))

    return seg_dsc_m, seg_dsc

def load_models(global_model, nodes, model_fname):
    global_model.load_state_dict(torch.load(model_fname)['global_model_state_dict'])
    for node_id in range(len(nodes)):
        nodes[node_id][0].load_state_dict(torch.load(model_fname)['local_model_{0:d}_state_dict'.format(node_id)])
        nodes[node_id][1][0].load_state_dict(torch.load(model_fname)['local_model_{0:d}_optimizer1'.format(node_id)])
        nodes[node_id][1][1].load_state_dict(torch.load(model_fname)['local_model_{0:d}_optimizer2'.format(node_id)])
        nodes[node_id][2][0].load_state_dict(torch.load(model_fname)['local_model_{0:d}_scheduler1'.format(node_id)])
        nodes[node_id][2][1].load_state_dict(torch.load(model_fname)['local_model_{0:d}_scheduler2'.format(node_id)])

def train():

    train_start_time = time.localtime()
    print("Start time: {start_time}\n".format(
            start_time=time.strftime("%Y-%m-%d %H:%M:%S", train_start_time)))
    time_stamp = time.strftime("%Y%m%d%H%M%S", train_start_time)
    
    # create directory for results storage
    store_dir = '{}/model_{}'.format(cfg['model_path'], time_stamp)
    loss_fn = '{}/loss.txt'.format(store_dir)
    val_result_path = '{}/results_val'.format(store_dir)
    os.makedirs(val_result_path, exist_ok=True)
    test_result_path = '{}/results_test'.format(store_dir)
    os.makedirs(test_result_path, exist_ok=True)

    print('Loading local data from each nodes ... \n')

    global_model, nodes, node_cls_flag = initialization()
    ood_nodes = initialize_ood_node()

    ma_val_acc = None
    best_val_acc = 0
    start_iter = 0
    acc_time = 0
    best_model_fn = '{0:s}/cp_commu_{1:04d}.pth.tar'.format(store_dir, 1)

    print()
    log_line = "Model: {}\nModel parameters: {}\nStart time: {}\nConfiguration:\n".format(
        global_model.module.description(), 
        sum(x.numel() for x in global_model.parameters()), 
        time.strftime("%Y-%m-%d %H:%M:%S", train_start_time))
    for cfg_key in cfg:
        log_line += ' --- {}: {}\n'.format(cfg_key, cfg[cfg_key])
    print(log_line)

    # training & validation phase
    for commu_t in range(start_iter, cfg['commu_times'], 1):
        
        t0 = time.perf_counter()

        train_loss = []
        for node_id, [local_model, optimizer, scheduler, node_name, node_weight, dl_train, dl_val, dl_test] in enumerate(nodes):
            
            print('Training ({0:d}/{1:d}) on Node: {2:s}\n'.format(commu_t+1, cfg['commu_times'], node_name))

            local_model.train()

            train_loss.append(train_local_model(local_model, optimizer, scheduler, dl_train, cfg['epoch_per_commu']))

        communication(global_model, nodes, node_cls_flag)

        t1 = time.perf_counter()
        epoch_t = t1 - t0
        acc_time += epoch_t
        print("Iteration time cost: {h:>02d}:{m:>02d}:{s:>02d}\n".format(
            h=int(epoch_t) // 3600, m=(int(epoch_t) % 3600) // 60, s=int(epoch_t) % 60))

        if commu_t % 1 == 0:
            seg_dsc_m, _ = eval_global_model(global_model, nodes, val_result_path, mode='val', commu_iters=commu_t)
            if ma_val_acc is None:
                ma_val_acc = seg_dsc_m
            else:
                ma_val_acc = seg_dsc_m

            loss_line = '{commu_iter:>04d}\t{train_loss:s}\t{seg_val_dsc:>8.6f}\t{ma_val_dsc:>8.6f}'.format(
                commu_iter=commu_t+1, train_loss='\t'.join(['%8.6f']*len(train_loss)) % tuple(train_loss), 
                seg_val_dsc=seg_dsc_m, ma_val_dsc=ma_val_acc)
            for node_id, [_, _, scheduler, _, _, _, _, _] in enumerate(nodes):
                loss_line += '\t{node_lr:>8.6f}'.format(node_lr=scheduler[0].get_last_lr()[0])
            loss_line += '\n'

            with open(loss_fn, 'a') as loss_file:
                loss_file.write(loss_line)

            # save best model
            if commu_t == 0 or ma_val_acc > best_val_acc:
                # remove former best model
                if os.path.exists(best_model_fn):
                    os.remove(best_model_fn)
                # save current best model
                best_val_acc = ma_val_acc
                best_model_fn = '{0:s}/cp_commu_{1:04d}.pth.tar'.format(store_dir, commu_t+1)
                best_model_cp = {
                            'commu_iter':commu_t,
                            'acc_time':acc_time,
                            'time_stamp':time_stamp,
                            'best_val_acc':best_val_acc,
                            'best_model_filename':best_model_fn,
                            'global_model_state_dict':global_model.state_dict()}
                for node_id, [local_model, optimizer, scheduler, _, _, _, _, _] in enumerate(nodes):
                    best_model_cp['local_model_{0:d}_state_dict'.format(node_id)] = local_model.state_dict()
                    best_model_cp['local_model_{0:d}_optimizer1'.format(node_id)] = optimizer[0].state_dict()
                    best_model_cp['local_model_{0:d}_optimizer2'.format(node_id)] = optimizer[1].state_dict()
                    best_model_cp['local_model_{0:d}_scheduler1'.format(node_id)] = scheduler[0].state_dict()
                    best_model_cp['local_model_{0:d}_scheduler2'.format(node_id)] = scheduler[1].state_dict()
                torch.save(best_model_cp, best_model_fn)
                print('Best model (communication iteration = {}) saved.\n'.format(commu_t+1))
        else:
            loss_line = '{commu_iter:>04d}\t{train_loss:s}'.format(
                commu_iter=commu_t+1, train_loss='\t'.join(['%8.6f']*len(train_loss)) % tuple(train_loss))
            for node_id, [_, _, scheduler, _, _, _, _, _] in enumerate(nodes):
                loss_line += '\t{node_lr:>8.6f}'.format(node_lr=scheduler[0].get_last_lr()[0])
            loss_line += '\n'
            with open(loss_fn, 'a') as loss_file:
                loss_file.write(loss_line)
    
    print("Total training time: {h:>02d}:{m:>02d}:{s:>02d}\n\n".format(
            h=int(acc_time) // 3600, m=(int(acc_time) % 3600) // 60, s=int(acc_time) % 60))

    # testing phase
    load_models(global_model, nodes, best_model_fn) # load the best-performence model according to validation results
    eval_global_model(global_model, nodes, test_result_path, mode='test', commu_iters=0) # test on the in-federation data
    eval_global_model(global_model, ood_nodes, test_result_path, mode='test-ood', commu_iters=0) # test on the out-of-federation data

    print("Finish time: {finish_time}\n\n".format(
            finish_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']
    train()