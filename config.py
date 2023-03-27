cfg = {}

cfg['cls_num'] = 5 # number of organs. the output of the segmentation network would have a channel number of (cfg['cls_num'] + 1)
cfg['gpu'] = '0,1,2,3' # to use multiple gpu: cfg['gpu'] = '0,1,2,3'
cfg['batch_size'] = 4 # batch size for training
cfg['test_batch_size'] = 4 # batch size for validation and testing
cfg['lr'] = 0.01 # base learning rate
cfg['model_path'] = '/fast/xux12/proj/fl-moseg/models' # path to store the trained models and testing results

cfg['rs_size'] = [256,256,32] # input image size: [W, H, D] ([x, y, z])
cfg['rs_spacing'] = [1.0,1.0,1.5] # input image spacing: [x, y, z]
cfg['rs_intensity'] = [-200.0, 400.0, -200.0, 600.0] # input image intensity normalization parameter: [min_intensity, max_intensity, min_intensity, max_intensity - min_intensity]

cfg['cpu_thread'] = 4 # multi-thread for data loading. zero means single thread.
cfg['commu_times'] = 400 # total number of communication rounds
cfg['epoch_per_commu'] = 1 # number of local training epochs

# map labels of different datasets to a uniform label map
cfg['label_map'] = {
    'LiTS':{1:1, 2:1},
    'KiTS':{1:2, 2:2},
    'pancreas':{1:3, 2:3},
    'spleen':{1:4},
    'AMOS':{6:1, 2:2, 3:2, 10:3, 1:4, 4:5},
    'BTCV':{6:1, 2:2, 3:2, 11:3, 1:4, 4:5},
}

# exclude any samples in the form of '[dataset_name, case_name]'
cfg['exclude_case'] = []

# client node list
# item format: 
# ['Node name', ['dataset1_name'], ['dataset1_datapath'], [num_training_sample, num_validation_sample, num_testing_sample]]
cfg['node_list'] = [
    ['Node-1', ['LiTS'], ['/fast/xux12/data/lits/training_downsampled'], [79,13,39]],
    ['Node-2', ['KiTS'], ['/fast/xux12/data/kits19/data_downsampled'], [126,21,63]],
    ['Node-3', ['pancreas'], ['/fast/xux12/data/decathlon/Task07_Pancreas'], [169,28,84]],
    ['Node-4', ['spleen'], ['/fast/xux12/data/spleen'], [24,5,12]],
    ['Node-5', ['AMOS'], ['/fast/xux12/data/AMOS22'], [120,20,60]],
]

# out-of-federation client node list
# (follow the same item format as above)
cfg['ood_node_list'] = [
    ['OoD-Node-1', ['BTCV'], ['/fast/xux12/data/btcv_downsampled'], [18,3,9]],
]