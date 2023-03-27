import torch
import torch.nn as nn
import numpy as np

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target, flag):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = self.smooth
        union = self.smooth
        if flag is None:
            pd = predict
            gt = target
            intersection += torch.sum(pd*gt)*2
            union += torch.sum(pd.pow(self.p) + gt.pow(self.p))
        else:
            for i in range(target.shape[0]):
                if flag[i,0] > 0:
                    pd = predict[i:i+1,:]
                    gt = target[i:i+1,:]
                    intersection += torch.sum(pd*gt)*2
                    union += torch.sum(pd.pow(self.p) + gt.pow(self.p))
        dice = intersection / union

        loss = 1 - dice
        return loss
        
class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=[], **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        if weight is not None:
            self.weight = weight / weight.sum()
        else:
            self.weight = None
        self.ignore_index = ignore_index

    def forward(self, predict, target, flag=None):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        total_loss_num = 0

        for c in range(target.shape[1]):
            if c not in self.ignore_index:
                dice_loss = dice(predict[:, c], target[:, c], flag)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[c]
                total_loss += dice_loss
                total_loss_num += 1

        if self.weight is not None:
            return total_loss
        elif total_loss_num > 0:
            return total_loss/total_loss_num
        else:
            return 0

def make_onehot(input, cls):
    oh_list = []
    for c in range(cls):
        tmp = torch.zeros_like(input)
        tmp[input==c] = 1        
        oh_list.append(tmp)
    oh = torch.cat(oh_list, dim=1)
    return oh

def merge_prob(prob, class_flag):
    bg_prob_list = []
    for c, class_exist in enumerate(class_flag):
        if c == 0 or class_exist == 0:
            bg_prob_list.append(prob[:,c:c+1,:])
    bg_prob = torch.sum(torch.cat(bg_prob_list, dim=1), dim=1, keepdim=True)
    merged_prob_list = [bg_prob]
    for c, class_exist in enumerate(class_flag):
        if c > 0 and class_exist > 0:
            merged_prob_list.append(prob[:,c:c+1,:])
    margin_prob = torch.cat(merged_prob_list, dim=1)
    return margin_prob

def merge_label(label, class_flag):
    merged_label = torch.zeros_like(label)
    cc = 0
    for c, class_exist in enumerate(class_flag):
        if c > 0 and class_exist > 0:
            merged_label[label == c] = cc + 1
            cc += 1
    return merged_label

def excluse_label(label):
    exclused_label = torch.zeros_like(label)
    exclused_label[label > 0] = 1
    return exclused_label

def marginal_loss(prob, target, class_flag):    
    margin_prob = merge_prob(prob, class_flag)
    margin_target = merge_label(target, class_flag)
    margin_log_prob = torch.log(torch.clamp(margin_prob, min=1e-4))
    ce_loss = nn.NLLLoss()
    l_ce = ce_loss(margin_log_prob, margin_target.squeeze(dim=1))
    margin_target_oh = make_onehot(margin_target, cls=np.sum(class_flag))
    dice_loss = DiceLoss()
    l_dice = dice_loss(margin_prob, margin_target_oh)
    return l_ce, l_dice

def exclusion_loss(prob, target, class_flag):
    epsilon = 1.0
    exclused_target = excluse_label(target)
    dice_loss = DiceLoss()
    l_ce = 0
    l_ce_num = 0
    l_dice = 0
    l_dice_num = 0
    for c, class_exist in enumerate(class_flag):
        if c == 0 or class_exist == 0:
            l_ce += torch.sum(exclused_target * torch.log(prob[:,c:c+1,:] + epsilon))
            l_ce_num += torch.sum(exclused_target)
            l_dice += 1 - dice_loss(prob[:,c:c+1,:], exclused_target)
            l_dice_num += 1
    if l_ce_num > 0:
        l_ce = l_ce / l_ce_num
    if l_dice_num > 0:
        l_dice = l_dice / l_dice_num
    return l_ce, l_dice

def dice_and_ce_loss(prob, target, class_flag):
    cls_num = np.sum(class_flag)
    l_ce = 0
    l_dice = 0
    loss_per_class = np.zeros(len(class_flag)-1, dtype=float)
    num_per_class = np.zeros(len(class_flag)-1, dtype=np.uint8)
    for c, class_exist in enumerate(class_flag):
        if c > 0 and class_exist > 0:
            bin_label = torch.zeros_like(target)
            bin_label[target == c] = 1
            bin_label_oh = make_onehot(bin_label, cls=2)

            bin_prob = torch.cat([1-prob[:,c:c+1,:], prob[:,c:c+1,:]], dim=1)            
            bin_prob_log = torch.log(torch.clamp(bin_prob, min=1e-4))

            ce_loss = nn.NLLLoss()
            l_ce += ce_loss(bin_prob_log, bin_label.squeeze(dim=1)) / (cls_num - 1)

            dice_loss = DiceLoss()
            l_dice_item = dice_loss(bin_prob, bin_label_oh) / (cls_num - 1)
            loss_per_class[c-1] = 1.0 - l_dice_item.item() * (cls_num - 1)
            num_per_class[c-1] = 1
            l_dice += l_dice_item

    return l_ce, l_dice, loss_per_class, num_per_class

def loc_dice_and_ce_loss(prob, target, label):
    bin_label = torch.zeros_like(target)
    bin_label[target == label] = 1
    bin_label_oh = make_onehot(bin_label, cls=2)

    bin_prob_log = torch.log(torch.clamp(prob, min=1e-4))

    ce_loss = nn.NLLLoss()
    l_ce = ce_loss(bin_prob_log, bin_label.squeeze(dim=1))

    dice_loss = DiceLoss()
    l_dice = dice_loss(prob, bin_label_oh)

    return l_ce, l_dice