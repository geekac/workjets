import torch
import numpy as np
import torch.nn as nn

from torch.nn import functional as F



__all__ = ['BCELoss', 'BalancedBCELoss',
           'DiceLoss', 'BCEDiceLoss', 'GeneralizedDiceLoss',
           'BinaryFocalLoss',
           'TverskyLoss']



# todo: 学习一下 https://github.com/LIVIAETS/surface-loss/blob/master/losses.py
# todo: 再总结一下 https://github.com/Hsuxu/Loss_ToolBox-PyTorch
# todo: 对比学习下 https://github.com/JunMa11/SegLoss/tree/master/losses_pytorch

"""
连接
https://www.cnblogs.com/banshaohuan/p/9493024.html
https://zhuanlan.zhihu.com/p/87959967?utm_source=wechat_session&utm_medium=social&utm_oi=999439652683530240
https://github.com/JunMa11/SegLoss

https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/focal_loss.py
https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py#L110

https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py

https://www.jianshu.com/p/30043bcc90b6
https://blog.csdn.net/CaiDaoqing/article/details/90457197
https://blog.csdn.net/m0_37477175/article/details/83004746#Dice_loss_70

https://blog.csdn.net/qq_42450404/article/details/92800575
https://blog.csdn.net/JMU_Ma/article/details/97533768


语义分割常用loss介绍及pytorch实现  https://blog.csdn.net/CaiDaoqing/article/details/90457197
                               https://blog.csdn.net/m0_37477175/article/details/83004746

"""

"""pytorch 实现dice loss的讨论：
    https://github.com/pytorch/pytorch/issues/1249

    nn.CrossEntropyLoss ->  多分类
    nn.BCELoss          ->  二分类
"""

# TODO: 规范 模型的输出为logits  重新写 `loss function`
# TODO: 如果使用softmax的目标函数，需要根据原始gt生成label数据  4分类问题， [0,1,2,4] ---> 4化为3


# import niftynet.layer.loss_segmentation as loss_segmentation
# loss_segmentation.dice()


# ----------------------      以二分类的方式进行处理                ----------------------------------- #
""" 模型的输入是(batch-size, 4, patch-size) 模型的输出是(batch-size, 3, patch-size)
    4是四种模态都使用，3是输出的标签种类数[1,2,4]  注意生成truth数据与值对应 !!!!
    idx=0标签值为1的病灶 idx=1标签值为2的病灶 idx=2标签值为4的病灶
"""

# 1 BCELoss
BCELoss = nn.BCEWithLogitsLoss
# nn.CrossEntropyLoss  todo: 注意 在pytorch中，多分类交叉熵 的输入不需要经过softmax处理
#       原因详见pytorch实现`多分类交叉熵`源代码  https://zhuanlan.zhihu.com/p/98785902


class BalancedBCELoss(nn.Module):    # 2 BalancedBCELoss
    def __init__(self):
        """
            参考链接 https://blog.csdn.net/qq_34914551/article/details/101644942
        """
        super(BalancedBCELoss, self).__init__()

    def forward(self, logits, target, reduction='mean'):
        pos = torch.eq(target, 1).float()
        neg = torch.eq(target, 0).float()

        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        num_total = num_pos + num_neg

        alpha_pos = num_neg / num_total
        alpha_neg = num_pos / num_total

        weights = alpha_pos * pos + alpha_neg * neg
        return F.binary_cross_entropy_with_logits(logits, target, weights, reduction=reduction)



class DiceLoss(nn.Module):    # 3 DiceLoss
    """
      医学图像分割之 Dice Loss  https://blog.csdn.net/JMU_Ma/article/details/97533768
      from https://blog.csdn.net/CaiDaoqing/article/details/90457197
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)   # 转换成概率

        N = target.size(0)
        smooth = 1

        input_flat = probs.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        dice_loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_loss.sum() / N

        return dice_loss



class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses

        from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py#L165
    """

    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, target):
        return self.alpha * self.bce(logits, target) + self.beta * self.dice(logits, target)



class GeneralizedDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon

    @staticmethod
    def flatten(tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
           (N, C, D, H, W) -> (C, N * D * H * W)
        """
        # number of channels
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.contiguous().view(C, -1)


    def forward(self, logits, target):
        probs = torch.sigmoid(logits)   # 转换成概率
        probs = self.flatten(probs)
        target = self.flatten(target)

        if probs.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            probs = torch.cat((probs, 1 - probs), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (probs * target).sum(-1)
        intersect = intersect * w_l

        denominator = (probs + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        # compute per channel Dice coefficient
        per_channel_dice = 2 * (intersect.sum() / denominator.sum())

        # average Dice score across all channels/classes
        generalized_dice_loss = 1. - torch.mean(per_channel_dice)
        return generalized_dice_loss



# class FocalLoss(nn.Module):     # 4 FocalLoss
#     """
#         使用下面的binaryFocalLoss!  原因见focal loss详解：
#         https://www.cnblogs.com/king-lps/p/9497836.html
#     """
#     def __init__(self, alpha=0.25, gamma=2, weight=None, ignore=255):
#         # from https://blog.csdn.net/CaiDaoqing/article/details/90457197
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.weight = weight
#         self.ignore = ignore
#         self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)
#
#     def forward(self, logits, target):
#         if self.ignore is not None:
#             mask = target != self.ignore
#             target = target[mask]
#             logits = logits[mask]
#
#         logpt = -self.bce_fn(logits, target)
#         pt = torch.exp(logpt)
#         focal_loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
#         return focal_loss



class BinaryFocalLoss(nn.Module):
    """
        link: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/focal_loss.py

    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=None, gamma=2, ignore_index=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if self.alpha is None:   # raw default: [1.0, 1.0]
            self.alpha = [0.25, 0.75]

        self.smooth = 1e-6
        assert self.reduction in ['none', 'mean', 'sum']

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2, ))
            assert self.alpha.shape[0] == 2, \
                'the `alpha` shape is not match the number of class'
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, probs), self.gamma) * torch.log(probs) * pos_mask
        neg_loss = -self.alpha[1] * torch.pow(probs, self.gamma) * torch.log(torch.sub(1.0, probs)) * neg_mask

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        if num_pos == 0:    # todo: 只检查了不含有病灶的  要是全是病灶呢? num_neg=0
            bianry_focal_loss = neg_loss
        else:
            bianry_focal_loss = pos_loss / num_pos + neg_loss / num_neg
        return bianry_focal_loss



class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=torch.sigmoid, batch_dice=False,
                 do_bg=True, smooth=1., square=False, alpha=0.3, beta=0.7):
        """
            from https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
            paper: https://arxiv.org/pdf/1706.05721.pdf

            修改apply_nonlin=torch.sigmoid 原来默认值为None
        """
        super(TverskyLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def sum_tensor(inp, axes, keepdim=False):
        # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
        axes = np.unique(axes).astype(int)
        if keepdim:
            for ax in axes:
                inp = inp.sum(int(ax), keepdim=True)
        else:
            for ax in sorted(axes, reverse=True):
                inp = inp.sum(int(ax))
        return inp


    def get_tp_fp_fn(self, net_output, gt, axes=None, mask=None, square=False):
        """
        net_output must be (b, c, x, y(, z)))
        gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
        if mask is provided it must have shape (b, 1, x, y(, z)))
        :param net_output:
        :param gt:
        :param axes:
        :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
        :param square: if True then fp, tp and fn will be squared before summation
        :return:
        """
        if axes is None:
            axes = tuple(range(2, len(net_output.size())))

        shp_x = net_output.shape
        shp_y = gt.shape

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        tp = net_output * y_onehot
        fp = net_output * (1 - y_onehot)
        fn = (1 - net_output) * y_onehot

        if mask is not None:
            tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
            fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
            fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

        if square:
            tp = tp ** 2
            fp = fp ** 2
            fn = fn ** 2

        tp = self.sum_tensor(tp, axes, keepdim=False)
        fp = self.sum_tensor(fp, axes, keepdim=False)
        fn = self.sum_tensor(fn, axes, keepdim=False)

        return tp, fp, fn


    def forward(self, logits, target, loss_mask=None):
        shp_x = logits.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            logits = self.apply_nonlin(logits)

        tp, fp, fn = self.get_tp_fp_fn(logits, target, axes, loss_mask, self.square)


        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return 1 - tversky



class BinaryTverskyLossV2(nn.Module):

    def __init__(self, alpha=0.3, beta=0.7, ignore_index=None, reduction='mean'):
        """
            todo: 和上面是一样的
            from https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/TverskyLoss/binarytverskyloss.py
        """
        """Dice loss of binary class
        Args:
            alpha: controls the penalty for false positives.
            beta: penalty for false negative.
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        Shapes:
            output: A tensor of shape [N, 1,(d,) h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """
        super(BinaryTverskyLossV2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.epsilon = 1e-6
        self.reduction = reduction
        s = self.beta + self.alpha
        if s != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s

    def forward(self, logits, target):
        batch_size = logits.size(0)

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            logits = logits.float().mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        probs = torch.sigmoid(logits).view(batch_size, -1)
        target = target.view(batch_size, -1)

        P_G = torch.sum(probs * target, 1)  # TP
        P_NG = torch.sum(probs * (1 - target), 1)  # FP
        NP_G = torch.sum((1 - probs) * target, 1)  # FN

        tversky_index = P_G / (P_G + self.alpha * P_NG + self.beta * NP_G + self.epsilon)

        loss = 1. - tversky_index
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss



class LovaszSoftmaxLoss(nn.Module):
    def __init__(self):
        """
            from https://blog.csdn.net/CaiDaoqing/article/details/90457197
            https://github.com/bermanmaxim/LovaszSoftmax/blob/7d48792d35a04d3167de488dd00daabbccd8334b/pytorch/lovasz_losses.py

        """
        super(LovaszSoftmaxLoss, self).__init__()

    def forward(self, logits, target):
        pass


    def lovasz_hinge(self, logits, labels, per_image=True, ignore=None):
        """
        Binary Lovasz hinge loss
          logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
          per_image: compute the loss per image instead of per batch
          ignore: void class id
        """
        if per_image:
            loss = mean(self.lovasz_hinge_flat(* self.flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                        for log, lab in zip(logits, labels))
        else:
            loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
        return loss

    @staticmethod
    def lovasz_hinge_flat(logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * Variable(signs))
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss

    @staticmethod
    def flatten_binary_scores(scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels


# todo: 结合 九、BCE + Dice loss  十、Dice + Focal loss

# ----------------------      以多分类的方式进行处理                ----------------------------------- #
""" 模型的输入是(batch-size, 4, patch-size) 模型的输出是(batch-size, 4, patch-size)
    这是一个四分类问题实际标签0/1/2/3 需要对truth进一步处理为，[0,0,0,1] 表示为第四类(one-hot 独热编码)
    3实际上在truth中为'4'
    ----------------------  分割问题不需要全连接层的
    F.softmax(logits, dim=1) 【注意】dim=1 !!!! 
"""




if __name__ == '__main__':
    loss_function1 = FocalLoss()
    loss_function2 = BinaryFocalLoss(alpha=None)
    loss_function3 = TverskyLoss()
    loss_function4 = BinaryTverskyLossV2()
    x = torch.rand((1, 3, 32, 32, 32))
    y = (torch.randn(1, 3, 32, 32, 32) > 0.5).float()

    loss1 = loss_function1(x, y)
    loss2 = loss_function2(x, y)
    loss3 = loss_function3(x, y)
    loss4 = loss_function4(x, y)
    print(loss1, loss2, loss3, loss4)





"""Discard - 丢弃不用 有参考价值的代码


class DiceLoss(nn.Module):      # todo:  需要重新实现
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, target, loss_type='jaccard'):
        smooth = 1e-5
        inse = torch.sum(inputs * target)

        if loss_type == 'jaccard':
            xl = torch.sum(inputs * inputs)
            r = torch.sum(target * target)
        elif loss_type == 'sorensen':
            xl = torch.sum(inputs)
            r = torch.sum(target)
        else:
            raise Exception("Unknown loss_type")

        dice = (2. * inse + smooth) / (xl + r + smooth)
        dice_loss = 1.0 - float(torch.mean(dice))
        return dice_loss

class FocalLoss(nn.Module):
    """ """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """ """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        print(one_hot_key)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLoss2(nn.Module):
    r""""""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """ """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss2, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = inputs

        class_mask = inputs.data.new(N, C).fill_(0)
        print(class_mask.size())
        ids = targets.view(-1, 1).cpu().long()
        print(ids.size())
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss



"""












